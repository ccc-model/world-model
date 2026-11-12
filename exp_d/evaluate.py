"""實驗 D 第 3 步：考 3D 世界模型。

(1) 探針 (linear probe)：從 M 的內部狀態，用線性迴歸讀出
    - agent 位置 (x, y)、朝向 (cos, sin)          → R²
    - 身邊 8 格是不是牆（3x3，絕對方位）          → 準確率
    - 整個迷宮內部 7x7 格局（看過的地方要記得）     → 準確率
    特徵：z_t（只有當下畫面）、LSTM h_t、Transformer h_t，以及未訓練的 LSTM 當對照。
(2) 原地轉圈：在轉圈回合，給轉圈前的真實畫面，之後只給「左轉」動作讓 M 做夢 80 步（約 4 圈）。
    真實畫面每 20 步回到原樣；夢是否也回到原樣？

用法：uv run python exp_d/evaluate.py
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_c.data import rollout_files
from exp_c.dream import dream, decode
from exp_c.models import VAE, make_m
from wm.depth import pick_device

KINDS = ["lstm", "transformer"]
R2_TARGETS = ("distance to wall ahead", "turned in last 20 steps", "position", "heading")


def to_tensor(x, dev):
    return torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def features(vae, models, d, dev, t_min):
    """回傳 {名稱: (T',D)}，只取 t >= t_min 的時間步。"""
    mu = torch.cat([vae.encode(to_tensor(d["obs"][i:i + 500], dev))[0] for i in range(0, len(d["obs"]), 500)])
    a = torch.tensor(d["act"], device=dev)
    out = {"z (current frame)": mu[t_min:]}
    for name, m in models.items():
        if isinstance(m, torch.nn.LSTM) or "lstm" in name:
            _, h, _ = m(mu[None], a[None])
            out[name] = h[0, t_min:]
        else:
            C = m.context
            hs = []
            for t0 in range(t_min, len(mu), 256):
                idx = torch.arange(t0, min(t0 + 256, len(mu)), device=dev)
                win = idx[:, None] - C + 1 + torch.arange(C, device=dev)[None]
                _, h, _ = m(mu[win], a[win])
                hs.append(h[:, -1])
            out[name] = torch.cat(hs)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def targets(d, t_min):
    pos, dr = d["pos"][t_min:], d["dir"][t_min:]
    # maze_layout[y, x]：1 = 可走、0 = 牆。外面一圈補牆，取 agent 周圍 3x3 去掉中心
    wall = np.pad(1 - d["layout"], 1, constant_values=1)
    cell = np.floor(pos).astype(int) + 1
    local = np.array([[wall[cy + dy, cx + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]
                      for cx, cy in cell], np.float32)
    full = np.repeat((1 - d["layout"]).reshape(1, -1).astype(np.float32), len(pos), 0)
    # 正前方到牆的距離（沿視線在格子地圖上 ray-march）：單張畫面就該看得出來的 3D 量
    S = np.arange(0, 9, 0.05)
    Q = np.floor(pos[:, None] + dr[:, None] * S[None, :, None]).astype(int) + 1
    Q = np.clip(Q, 0, wall.shape[0] - 1)
    ahead = S[np.argmax(wall[Q[..., 1], Q[..., 0]] == 1, axis=1)][:, None]
    # 過去 20 步轉了多少：要靠記憶（路徑積分）才知道
    ang = np.arctan2(d["dir"][:, 1], d["dir"][:, 0])
    dth = ang[t_min:] - ang[t_min - 20:len(ang) - 20]
    turned = np.stack([np.cos(dth), np.sin(dth)], 1)
    return {"distance to wall ahead": ahead, "turned in last 20 steps": turned,
            "position": pos, "heading": dr, "local walls (8)": local, "maze layout (9x9)": full}


def ridge(Xtr, Ytr, Xte, lam=1000.0):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - m) / s, (Xte - m) / s
    A, B = np.hstack([A, np.ones((len(A), 1))]), np.hstack([B, np.ones((len(B), 1))])
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
    return B @ W


def score(name, pred, y, ytr):
    if name in R2_TARGETS:
        return 1 - ((pred - y) ** 2).sum() / ((y - ytr.mean(0)) ** 2).sum()          # R²
    return float(((pred > 0.5) == (y > 0.5)).mean())                                  # 準確率


def probe(vae, models, dev, args):
    tr_files = rollout_files(args.data, "train")[:args.probe_train]
    te_files = rollout_files(args.data, "test")
    F, T = {"train": [], "test": []}, {"train": [], "test": []}
    for split, files in (("train", tr_files), ("test", te_files)):
        for f in files:
            d = dict(np.load(f))
            F[split].append(features(vae, models, d, dev, args.t_min))
            T[split].append(targets(d, args.t_min))
    cat = lambda L, k: np.concatenate([x[k] for x in L])
    res = {}
    for tk in T["train"][0]:
        ytr, yte = cat(T["train"], tk), cat(T["test"], tk)
        row = {}
        if tk not in R2_TARGETS:
            row["baseline (majority)"] = float(((ytr.mean(0) > 0.5) == (yte > 0.5)).mean())
        for fk in F["train"][0]:
            row[fk] = float(score(tk, ridge(cat(F["train"], fk), ytr, cat(F["test"], fk)), yte, ytr))
        res[tk] = row
    return res


@torch.no_grad()
def turn_test(vae, models, dev, args):
    files = sorted(glob.glob(f"{args.data}/turn_*.npz"))
    real, ctxZ, acts, starts = [], [], [], []
    for f in files:
        d = np.load(f)
        s = int(d["turn_start"])
        if len(d["obs"]) < s + args.turn:
            continue
        mu = torch.cat([vae.encode(to_tensor(d["obs"][i:i + 500], dev))[0] for i in range(0, s + args.turn, 500)])
        real.append(d["obs"][s:s + args.turn]); ctxZ.append(mu[:s]); acts.append(d["act"][:s + args.turn])
        starts.append(d["obs"][s - 1])
    real, starts = np.array(real), np.array(starts)
    Z = torch.stack(ctxZ)                                     # 所有轉圈回合的 walk 長度相同
    A = torch.tensor(np.array(acts), device=dev)
    R = torch.from_numpy(real).to(dev).float() / 255
    S = torch.from_numpy(starts).to(dev).float()[:, None] / 255
    curves = {}
    curves["real vs turn-start frame"] = ((R - S) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
    curves["copy last frame"] = curves["real vs turn-start frame"]           # 凍結畫面 = 起始畫面
    imgs = {}
    for k, m in models.items():
        zd = dream(m, k, Z, A, 0)
        img = decode(vae, zd)
        curves[k] = ((img - R) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
        curves[k + " vs turn-start"] = ((img - S) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
        imgs[k] = img.cpu().numpy()
        imgs[k + "_sample"] = decode(vae, dream(m, k, Z[:3], A[:3], 1.0)).cpu().numpy()
    return curves, real, imgs, len(real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/memmaze")
    ap.add_argument("--prefix", default="exp_d")
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--probe-train", type=int, default=30, help="用幾個訓練回合擬合探針")
    ap.add_argument("--t-min", type=int, default=100, help="探針只看第 t_min 步之後（已探索一段時間）")
    ap.add_argument("--turn", type=int, default=80)
    args = ap.parse_args()
    dev = pick_device()
    torch.manual_seed(0)
    P = args.prefix

    vae = VAE().to(dev).eval()
    vae.load_state_dict(torch.load(f"out/{P}_vae.pt", map_location=dev))
    models = {}
    for k in KINDS:
        m = make_m(k, a_dim=6, context=args.context).to(dev).eval()
        m.load_state_dict(torch.load(f"out/{P}_m_{k}.pt", map_location=dev))
        models[k] = m
    probe_models = dict(models)
    probe_models["lstm (untrained)"] = make_m("lstm", a_dim=6).to(dev).eval()

    # (1) 探針
    res = probe(vae, probe_models, dev, args)
    print("\nLinear probe (test episodes)  position/heading: R²; walls/layout: accuracy")
    cols = ["baseline (majority)", "z (current frame)", "lstm (untrained)", "lstm", "transformer"]
    print(f"{'target':20s}" + "".join(f"{c[:18]:>20s}" for c in cols))
    for tk, row in res.items():
        print(f"{tk:20s}" + "".join(f"{row[c]:20.3f}" if c in row else f"{'-':>20s}" for c in cols))

    # (2) 原地轉圈
    curves, real, imgs, n = turn_test(vae, models, dev, args)
    print(f"\nTurn-in-place dream test on {n} episodes (pixel MSE vs real at step k)")
    for k in (5, 10, 20, 40, 60, 80):
        print(f"  step {k:2d}: " + "  ".join(f"{c}={curves[c][k - 1]:.4f}" for c in ["copy last frame", "lstm", "transformer"]))
    json.dump(dict(probe=res, turn={c: v.tolist() for c, v in curves.items()}), open(f"out/{P}_eval.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, args.turn + 1)
    ax.plot(x, curves["copy last frame"], "--", color="gray", label="copy last frame (= real vs start)")
    for k, c in zip(KINDS, ["tab:green", "tab:red"]):
        ax.plot(x, curves[k], color=c, label=f"{k} dream vs real")
    for v in range(20, args.turn + 1, 20):
        ax.axvline(v, color="k", alpha=0.15)
    ax.axvline(args.context - 0.5, color="tab:red", ls=":", alpha=0.6)
    ax.text(args.context, ax.get_ylim()[1] * 0.95, " transformer context", color="tab:red", fontsize=8, va="top")
    ax.set_xlabel("dream step (turning left, ~18.4°/step; grey lines = full circles)")
    ax.set_ylabel("pixel MSE"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Turn in place inside the dream: does the world come back?")
    fig.tight_layout(); fig.savefig(f"out/{P}_turn_error.png", dpi=110); plt.close(fig)

    # 截圖：第 0 集，各步
    steps = [1, 5, 10, 15, 20, 40, 60, 80]
    rows = [real[0][[s - 1 for s in steps]], imgs["lstm"][0][[s - 1 for s in steps]], imgs["transformer"][0][[s - 1 for s in steps]]]
    fig, ax = plt.subplots(3, len(steps), figsize=(2 * len(steps), 6.4))
    for r, (name, row) in enumerate(zip(["real", "LSTM dream", "Transformer dream"], rows)):
        for c, im in enumerate(row):
            ax[r, c].imshow(np.clip(im, 0, 255) if im.dtype == np.uint8 else np.clip(im, 0, 1))
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(f"step {steps[c]}")
        ax[r, 0].set_ylabel(name)
    fig.tight_layout(); fig.savefig(f"out/{P}_turn_frames.png", dpi=90); plt.close(fig)

    # 影片：3 集，取樣夢境
    S = 192
    vw = cv2.VideoWriter(f"out/{P}_turn.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 8, (S * 3, (S + 20) * 3))
    for t in range(args.turn):
        rows = []
        for j in range(3):
            tiles = [real[j][t], (imgs["lstm_sample"][j][t] * 255).astype(np.uint8), (imgs["transformer_sample"][j][t] * 255).astype(np.uint8)]
            bar = np.zeros((20, S * 3, 3), np.uint8)
            for i, name in enumerate(["real", "LSTM dream", "Transformer dream"]):
                cv2.putText(bar, f"{name} | turn step {t + 1}", (5 + i * S, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            rows += [bar, np.concatenate([cv2.resize(x, (S, S), interpolation=cv2.INTER_NEAREST) for x in tiles], 1)]
        vw.write(cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"→ out/{P}_eval.json, out/{P}_turn_error.png, out/{P}_turn_frames.png, out/{P}_turn.mp4")


if __name__ == "__main__":
    main()
