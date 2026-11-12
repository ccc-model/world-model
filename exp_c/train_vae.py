"""實驗 C 第 2 步：訓練 VAE（V），再把所有回合編碼成 z 序列存起來給 M 用。

用法：uv run python exp_c/train_vae.py --epochs 10
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_c.data import load_frames, rollout_files
from exp_c.models import VAE
from wm.depth import pick_device


def to_tensor(x, dev):
    return torch.from_numpy(x).to(dev).permute(0, 3, 1, 2).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--kl-tol", type=float, default=0.5, help="每維 KL 免費額度（原論文 0.5）")
    args = ap.parse_args()
    dev = pick_device()
    os.makedirs("out", exist_ok=True)

    X = load_frames(rollout_files(split="train"))
    Xte = load_frames(rollout_files(split="test"))
    print(f"train {len(X)} frames, test {len(Xte)}, device {dev}")
    vae = VAE().to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(args.epochs):
        perm = np.random.permutation(len(X))
        rl = kl = 0.0
        for i in range(0, len(X) - args.bs + 1, args.bs):
            x = to_tensor(X[perm[i:i + args.bs]], dev)
            rec, mu, logvar = vae(x)
            r = ((rec - x) ** 2).sum((1, 2, 3)).mean()
            k = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
            loss = r + torch.clamp(k, min=args.kl_tol * mu.shape[1])
            opt.zero_grad(); loss.backward(); opt.step()
            rl += r.item(); kl += k.item()
        n = len(X) // args.bs
        with torch.no_grad():
            x = to_tensor(Xte[::20], dev)
            te = ((vae.decode(vae.encode(x)[0]) - x) ** 2).mean().item()
        print(f"  epoch {ep}  recon {rl / n:.1f}  KL {kl / n:.1f}  test MSE/pixel {te:.5f}  {time.time() - t0:.0f}s")
    torch.save(vae.state_dict(), "out/exp_c_vae.pt")

    # 重建對照圖
    with torch.no_grad():
        idx = np.linspace(0, len(Xte) - 1, 8).astype(int)
        x = to_tensor(Xte[idx], dev)
        rec = vae.decode(vae.encode(x)[0]).permute(0, 2, 3, 1).cpu().numpy()
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
    for j in range(8):
        ax[0, j].imshow(Xte[idx[j]]); ax[1, j].imshow(rec[j])
        ax[0, j].axis("off"); ax[1, j].axis("off")
    ax[0, 0].set_title("real", loc="left"); ax[1, 0].set_title("VAE (z=32)", loc="left")
    fig.tight_layout(); fig.savefig("out/exp_c_vae_recon.png", dpi=90); plt.close(fig)

    # 把每個回合編碼成 (mu, logvar) 序列
    zs = {}
    with torch.no_grad():
        for split in ("train", "test"):
            for f in rollout_files(split=split):
                d = np.load(f)
                mu, lv = vae.encode(to_tensor(d["obs"], dev))
                zs[os.path.basename(f)] = dict(mu=mu.cpu().numpy(), logvar=lv.cpu().numpy(), act=d["act"], split=split)
    np.save("out/exp_c_z.npy", zs, allow_pickle=True)
    print(f"done in {time.time() - t0:.0f}s → out/exp_c_vae.pt, out/exp_c_z.npy, out/exp_c_vae_recon.png")


if __name__ == "__main__":
    main()
