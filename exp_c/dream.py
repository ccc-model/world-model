"""實驗 C 第 4 步：讓 M 「做夢」。

給 16 幀真實畫面當開頭，之後完全不看真實畫面，只用真實的動作序列，
讓 M 自己一步步想像下一個 z，再用 VAE 解碼成畫面。輸出：
  - 開環預測誤差曲線：想像畫面 vs 真實畫面的像素 MSE 隨步數變化（取期望值，溫度 0）
  - 夢境影片：真實 | LSTM 夢 | Transformer 夢（溫度 1，取樣）

用法：uv run python exp_c/dream.py
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_c.data import rollout_files
from exp_c.models import VAE, MDNHead, make_m
from wm.depth import pick_device

KINDS = ["lstm", "transformer"]


@torch.no_grad()
def dream(model, kind, z_ctx, actions, temperature):
    """z_ctx: (B,C,32) 真實開頭；actions: (B,C+H,3)。回傳想像的 (B,H,32)。"""
    C, H = z_ctx.shape[1], actions.shape[1] - z_ctx.shape[1]
    out = []
    if kind == "lstm":
        params, _, state = model(z_ctx, actions[:, :C])
        last = lambda p: tuple(x[:, -1:] for x in p)
        for t in range(H):
            z = MDNHead.sample(last(params), temperature)
            out.append(z)
            if t < H - 1:
                params, _, state = model(z, actions[:, C + t:C + t + 1], state)
    else:
        zs = [z_ctx[:, i:i + 1] for i in range(C)]
        for t in range(H):
            lo = max(0, C + t - model.context)
            params, _, _ = model(torch.cat(zs[lo:], 1), actions[:, lo:C + t])
            z = MDNHead.sample(tuple(x[:, -1:] for x in params), temperature)
            zs.append(z); out.append(z)
    return torch.cat(out, 1)


def decode(vae, z):
    B, T, _ = z.shape
    img = vae.decode(z.reshape(B * T, -1)).permute(0, 2, 3, 1)
    return img.reshape(B, T, 64, 64, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--start", type=int, default=60, help="跳過開場的鏡頭拉近動畫")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    dev = pick_device()
    torch.manual_seed(0)

    vae = VAE().to(dev).eval()
    vae.load_state_dict(torch.load("out/exp_c_vae.pt", map_location=dev))
    models = {}
    for k in KINDS:
        m = make_m(k).to(dev).eval()
        m.load_state_dict(torch.load(f"out/exp_c_m_{k}.pt", map_location=dev))
        models[k] = m

    # 測試回合：每回合取幾個起點
    obs, Z, A = [], [], []
    L = args.ctx + args.horizon
    for f in rollout_files(split="test"):
        d = np.load(f)
        for s in range(args.start, len(d["obs"]) - L + 1, 100):
            obs.append(d["obs"][s:s + L]); A.append(d["act"][s:s + L])
    obs = np.array(obs)
    x = torch.from_numpy(obs.reshape(-1, 64, 64, 3)).to(dev).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        Z = torch.cat([vae.encode(x[i:i + 512])[0] for i in range(0, len(x), 512)]).view(len(obs), L, -1)
    A = torch.tensor(np.array(A), device=dev)
    real = torch.from_numpy(obs[:, args.ctx:]).to(dev).float() / 255
    print(f"{len(obs)} test clips, context {args.ctx}, horizon {args.horizon}")

    # 1) 開環誤差曲線（溫度 0 = 期望值）
    curves, speed = {}, {}
    with torch.no_grad():
        recon = decode(vae, Z[:, args.ctx:])
        curves["VAE recon (upper bound)"] = ((recon - real) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
        copy = torch.from_numpy(obs[:, args.ctx - 1:args.ctx]).to(dev).float() / 255
        curves["copy last frame"] = ((copy - real) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
        for k, m in models.items():
            t0 = time.time()
            zd = dream(m, k, Z[:, :args.ctx], A, 0)
            if dev == "mps":
                torch.mps.synchronize()
            speed[k] = len(obs) * args.horizon / (time.time() - t0)
            curves[k] = ((decode(vae, zd) - real) ** 2).mean((0, 2, 3, 4)).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, c in curves.items():
        ax.plot(np.arange(1, len(c) + 1), c, label=k, ls="--" if "VAE" in k or "copy" in k else "-")
    ax.set_xlabel("steps into the dream (no real frames)"); ax.set_ylabel("pixel MSE vs real")
    ax.set_title("Open-loop prediction error (temperature 0)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("out/exp_c_dream_error.png", dpi=110); plt.close(fig)
    summary = {k: {h: float(c[h - 1]) for h in (1, 10, 30, 100) if h <= len(c)} for k, c in curves.items()}
    summary["dream_steps_per_sec"] = speed
    json.dump(summary, open("out/exp_c_dream.json", "w"), indent=2)
    for k, v in summary.items():
        print(f"  {k:24s} {v}")

    # 2) 夢境影片（取樣），挑 3 段
    picks = np.linspace(0, len(obs) - 1, 3).astype(int)
    with torch.no_grad():
        dreams = {k: decode(vae, dream(m, k, Z[picks, :args.ctx], A[picks], args.temperature)).cpu().numpy()
                  for k, m in models.items()}
    S = 192
    vw = cv2.VideoWriter("out/exp_c_dream.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (S * 3, S * 3 + 20 * 3))
    for t in range(L):
        rows = []
        for j, p in enumerate(picks):
            tiles = [obs[p, t]]
            for k in KINDS:
                tiles.append(obs[p, t] if t < args.ctx else (dreams[k][j, t - args.ctx] * 255).astype(np.uint8))
            row = np.concatenate([cv2.resize(im, (S, S), interpolation=cv2.INTER_NEAREST) for im in tiles], 1)
            bar = np.zeros((20, S * 3, 3), np.uint8)
            label = "context (real)" if t < args.ctx else f"dream step {t - args.ctx + 1}"
            for i, name in enumerate(["real", "LSTM dream", "Transformer dream"]):
                cv2.putText(bar, f"{name} | {label}", (5 + i * S, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            rows += [bar, row]
        vw.write(cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    vw.release()
    print("→ out/exp_c_dream_error.png, out/exp_c_dream.mp4")


if __name__ == "__main__":
    main()
