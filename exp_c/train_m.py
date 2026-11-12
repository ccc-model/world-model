"""實驗 C 第 3 步：訓練 M（預測下一個 z），LSTM 與 Transformer 用相同資料、相同步數。

用法：uv run python exp_c/train_m.py --kind lstm
      uv run python exp_c/train_m.py --kind transformer
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from exp_c.models import MDNHead, make_m
from wm.depth import pick_device


def batches(rollouts, bs, L, rng):
    """隨機抽 bs 段長度 L+1 的連續 (z, a)；z 從 VAE 的後驗分布取樣。"""
    keys = rng.choice(len(rollouts), bs)
    Z, A = [], []
    for k in keys:
        r = rollouts[k]
        s = rng.integers(0, len(r["mu"]) - L - 1)
        mu, lv = r["mu"][s:s + L + 1], r["logvar"][s:s + L + 1]
        Z.append(mu + rng.standard_normal(mu.shape) * np.exp(0.5 * lv))
        A.append(r["act"][s:s + L + 1])
    return np.array(Z, np.float32), np.array(A, np.float32)


def test_nll(model, rollouts, L, dev):
    tot, n = 0.0, 0
    with torch.no_grad():
        for r in rollouts:
            for s in range(0, len(r["mu"]) - L - 1, L):
                z = torch.tensor(r["mu"][None, s:s + L + 1], device=dev)
                a = torch.tensor(r["act"][None, s:s + L + 1], device=dev)
                params, _, _ = model(z[:, :-1], a[:, :-1])
                tot += MDNHead.nll(params, z[:, 1:]).item(); n += 1
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--context", type=int, default=32, help="Transformer 的 context 長度")
    ap.add_argument("--prefix", default="exp_c")
    args = ap.parse_args()
    dev = pick_device()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    zs = np.load(f"out/{args.prefix}_z.npy", allow_pickle=True).item()
    train = [v for v in zs.values() if v["split"] == "train"]
    test = [v for v in zs.values() if v["split"] == "test"]
    model = make_m(args.kind, a_dim=train[0]["act"].shape[1], context=args.context).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    print(f"{args.kind}: {n_params / 1e3:.0f}K params, device {dev}")
    t0 = time.time()
    for step in range(args.steps):
        Z, A = batches(train, args.bs, args.seq, rng)
        z, a = torch.tensor(Z, device=dev), torch.tensor(A, device=dev)
        params, _, _ = model(z[:, :-1], a[:, :-1])
        loss = MDNHead.nll(params, z[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  train NLL {loss.item():.3f}  {time.time() - t0:.0f}s")
    train_time = time.time() - t0
    nll = test_nll(model, test, args.seq, dev)
    print(f"test NLL {nll:.3f}  train time {train_time:.0f}s")
    torch.save(model.state_dict(), f"out/{args.prefix}_m_{args.kind}.pt")
    json.dump(dict(kind=args.kind, params=n_params, test_nll=nll, train_time=train_time, steps=args.steps),
              open(f"out/{args.prefix}_m_{args.kind}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
