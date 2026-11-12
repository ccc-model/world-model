"""實驗 B1：在 KITTI Tracking 的 3D 真值軌跡上，訓練小 GRU 預測未來，和物理基準比較。

用法：
  uv run python exp_b/train.py                  # 真值輸入（沒有感知噪音）
  uv run python exp_b/train.py --noise          # 輸入加上單眼深度噪音（模擬實驗 A）
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_b.kitti import list_sequences, load_sequence, make_windows

GROUP = {"Car": 0, "Van": 0, "Truck": 0, "Tram": 0, "Pedestrian": 1, "Person_sitting": 1, "Cyclist": 2}
GROUP_NAMES = ["vehicle", "pedestrian", "cyclist"]


def canon(obs):
    """平移到最後一個觀測點、旋轉成「最近移動方向朝 +x」。回傳 (正規化後的 obs, R, origin)。"""
    origin = obs[:, -1]
    d = obs[:, -1] - obs[:, -4]
    ang = np.where(np.linalg.norm(d, axis=1) > 0.2, np.arctan2(d[:, 1], d[:, 0]), 0.0)
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.stack([np.stack([c, -s], 1), np.stack([s, c], 1)], 1)       # (B,2,2)
    rel = obs - origin[:, None]
    return np.einsum("bij,btj->bti", R, rel), R, origin


def uncanon(pred, R, origin):
    return np.einsum("bji,btj->bti", R, pred) + origin[:, None]       # R^T


class GRUPredictor(nn.Module):
    def __init__(self, n_pred, hidden=64):
        super().__init__()
        self.n_pred = n_pred
        self.gru = nn.GRU(2 + 3, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_pred * 2))

    def forward(self, obs_c, group):
        """obs_c: (B,T,2) 正規化後的過去位置；group: (B,) 類別。輸出 (B,n_pred,2)。"""
        delta = torch.diff(obs_c, dim=1, prepend=obs_c[:, :1])
        onehot = nn.functional.one_hot(group, 3).float()[:, None].expand(-1, obs_c.shape[1], -1)
        h, _ = self.gru(torch.cat([delta, onehot], -1))
        return self.head(h[:, -1]).view(-1, self.n_pred, 2)


def baselines(obs, n_pred, dt):
    steps = np.arange(1, n_pred + 1)[None, :, None]
    last = obs[:, -1:]
    v_last = (obs[:, -1] - obs[:, -2])[:, None]
    # 用過去所有點做最小平方直線擬合求速度（等於一個簡單的平滑器）
    t = np.arange(obs.shape[1]) - (obs.shape[1] - 1)
    A = np.stack([t, np.ones_like(t)], 1)
    coef = np.einsum("ij,bjk->bik", np.linalg.pinv(A), obs)          # (B,2,2): slope, intercept
    v_fit, p_fit = coef[:, 0:1], coef[:, 1:2]
    return {"static": np.repeat(last, n_pred, 1),
            "cv_last": last + v_last * steps,
            "cv_fit": p_fit + v_fit * steps}


def metrics(pred, gut, mask):
    d = np.linalg.norm(pred - gut, axis=-1)
    return {"ADE": float(d[mask].mean()), "FDE": float(d[mask, -1].mean()), "n": int(mask.sum())}


def load_split(root, seqs, args, rng):
    W = []
    for s in seqs:
        noise = (0.02, 0.10) if args.noise else None
        W += make_windows(load_sequence(root, s, noise, rng), args.obs, args.pred, seq=s)
    obs = np.array([w["obs"] for w in W])
    fut = np.array([w["fut"] for w in W])
    grp = np.array([GROUP[w["cls"]] for w in W])
    return W, obs, fut, grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kitti/training")
    ap.add_argument("--obs", type=int, default=10, help="過去幾幀（KITTI 10 fps）")
    ap.add_argument("--pred", type=int, default=10, help="預測幾幀")
    ap.add_argument("--test-seqs", default="16-20")
    ap.add_argument("--noise", action="store_true", help="輸入加單眼深度噪音 (σxy=0.02Z, σz=0.10Z)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    a, b = map(int, args.test_seqs.split("-"))
    seqs = list_sequences(args.root)
    test = [s for s in seqs if a <= s <= b]
    train = [s for s in seqs if s not in test]
    t0 = time.time()
    _, Xtr, Ytr, Gtr = load_split(args.root, train, args, rng)
    Wte, Xte, Yte, Gte = load_split(args.root, test, args, rng)
    Xclean = Xte
    if args.noise:
        # 評估答案用無噪音的真值：重新載入乾淨的軌跡
        Wc, Xclean, Yc, _ = load_split(args.root, test, argparse.Namespace(**{**vars(args), "noise": False}), rng)
        assert [(w["seq"], w["tid"], w["frame"]) for w in Wc] == [(w["seq"], w["tid"], w["frame"]) for w in Wte]
        Yte = Yc
    print(f"train {len(Xtr)} windows ({len(train)} seqs), test {len(Xte)} windows ({len(test)} seqs), "
          f"load {time.time() - t0:.1f}s")

    # --- 訓練 GRU（在正規化座標上） ---
    oc, R, o = canon(Xtr)
    yc = np.einsum("bij,btj->bti", R, Ytr - o[:, None])
    X, Y, G = (torch.tensor(oc, dtype=torch.float32), torch.tensor(yc, dtype=torch.float32), torch.tensor(Gtr))
    model = GRUPredictor(args.pred)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = (model(X[idx], G[idx]) - Y[idx]).norm(dim=-1).mean()     # 直接最小化 ADE
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train ADE {tot / len(X):.3f} m")
    print(f"train time {time.time() - t0:.1f}s")

    # --- 評估 ---
    oc, R, o = canon(Xte)
    with torch.no_grad():
        pc = model(torch.tensor(oc, dtype=torch.float32), torch.tensor(Gte)).numpy()
    preds = baselines(Xte, args.pred, 0.1)
    preds["gru"] = uncanon(pc, R, o)
    moving = np.linalg.norm(Yte[:, -1] - Xclean[:, -1], axis=1) > 0.5 * args.pred / 10   # 未來平均速度 > 0.5 m/s
    subsets = {"all": np.ones(len(Xte), bool), "moving": moving}
    for g, name in enumerate(GROUP_NAMES):
        subsets[f"moving_{name}"] = moving & (Gte == g)
    res = {k: {m: metrics(p, Yte, mask) for m, p in preds.items()} for k, mask in subsets.items() if mask.any()}

    tag = f"exp_b_{'noise' if args.noise else 'gt'}_p{args.pred}"
    os.makedirs("out", exist_ok=True)
    print(f"\nhorizon {args.pred / 10:.1f}s, input={'noisy' if args.noise else 'ground truth'}")
    print(f"{'subset':20s}" + "".join(f"{m:>16s}" for m in preds))
    for k, r in res.items():
        print(f"{k + ' (' + str(r['gru']['n']) + ')':20s}" + "".join(f"{r[m]['ADE']:7.2f}/{r[m]['FDE']:<7.2f} " for m in preds))
    print("(ADE/FDE, meters)")
    json.dump(res, open(f"out/{tag}_metrics.json", "w"), indent=2)
    torch.save(model.state_dict(), f"out/{tag}_gru.pt")

    # --- 畫幾個「會動、而且 GRU 和 cv_fit 差最多」的例子 ---
    gain = (np.linalg.norm(preds["cv_fit"] - Yte, axis=-1).mean(1) - np.linalg.norm(preds["gru"] - Yte, axis=-1).mean(1))
    gain[~moving] = -np.inf
    picks = list(np.argsort(-gain)[:4]) + list(rng.choice(np.where(moving)[0], 4, replace=False))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, i in zip(axes.flat, picks):
        ax.plot(*Xte[i].T, "k.-", label="past")
        ax.plot(*Yte[i].T, "g.-", label="future (GT)")
        for m, c in [("static", "gray"), ("cv_fit", "tab:blue"), ("gru", "tab:red")]:
            ax.plot(*np.vstack([Xte[i][-1:], preds[m][i]]).T, "--", color=c, label=m)
        ax.set_title(f"seq{Wte[i]['seq']} #{Wte[i]['tid']} {Wte[i]['cls']}")
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"KITTI tracking, BEV world frame (m). top: GRU wins most, bottom: random moving. input={'noisy' if args.noise else 'GT'}")
    fig.tight_layout(); fig.savefig(f"out/{tag}_examples.png", dpi=100); plt.close(fig)
    print("→", f"out/{tag}_examples.png")


if __name__ == "__main__":
    main()
