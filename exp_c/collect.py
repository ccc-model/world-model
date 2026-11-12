"""實驗 C 第 1 步：用隨機策略在 CarRacing 收集畫面與動作。

動作用平滑的隨機過程（方向盤是 Ornstein-Uhlenbeck，油門偏向前進），
比純隨機更容易開出各種彎道畫面。每回合存成 data/carracing/rollout_XXXX.npz：
  obs (T,64,64,3) uint8、act (T,3) float32、rew (T,) float32

用法：uv run python exp_c/collect.py --n 300 --steps 400 --workers 4
"""
import argparse, os, time
from multiprocessing import Pool

import cv2
import gymnasium as gym
import numpy as np


def rollout(args):
    idx, steps, out = args
    rng = np.random.default_rng(idx)
    env = gym.make("CarRacing-v3")
    o, _ = env.reset(seed=int(idx))
    obs, act, rew = [], [], []
    steer, gas = 0.0, 0.5
    for t in range(steps):
        steer = np.clip(0.8 * steer + rng.normal(0, 0.35), -1, 1)
        gas = np.clip(0.9 * gas + 0.1 * rng.uniform(0, 1) + rng.normal(0, 0.05), 0, 1)
        brake = 0.8 if rng.random() < 0.05 else 0.0
        a = np.array([steer, gas, brake], np.float32)
        obs.append(cv2.resize(o, (64, 64), interpolation=cv2.INTER_AREA))
        act.append(a)
        o, r, term, trunc, _ = env.step(a)
        rew.append(r)
        if term or trunc:
            break
    env.close()
    np.savez_compressed(os.path.join(out, f"rollout_{idx:04d}.npz"),
                        obs=np.array(obs), act=np.array(act), rew=np.array(rew, np.float32))
    return len(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/carracing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0, total = time.time(), 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(rollout, [(i, args.steps, args.out) for i in range(args.n)])):
            total += n
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.n} rollouts, {total} frames, {time.time() - t0:.0f}s")
    print(f"done: {total} frames in {time.time() - t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
