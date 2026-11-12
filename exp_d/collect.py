"""實驗 D 第 1 步：在 Memory Maze 9x9（第一人稱 3D 迷宮）收集資料。

兩種回合：
  - 一般回合 rollout_XXXX.npz：隨機探索 1000 步（動作會持續 1–4 步），供訓練 V/M 與探針。
  - 轉圈回合 turn_XXXX.npz：隨機走 --walk 步、停 3 步，再原地左轉 --turn 步（每步約 18.4°，20 步約一圈）。
    真實畫面轉一圈會回到原來的樣子，用來考「夢裡轉一圈，世界還在不在」。

每回合存 obs (T,64,64,3) uint8、act (T,6) one-hot、pos (T,2)、dir (T,2)、layout (9,9)。
動作：0 不動、1 前進、2 左轉、3 右轉、4 前進左轉、5 前進右轉。

用法：MUJOCO_GL=glfw uv run python exp_d/collect.py
"""
import argparse, os, time
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np

N_ACT = 6
PROBS = np.array([0.0, 0.5, 0.15, 0.15, 0.1, 0.1])


def make_env(seed):
    from memory_maze import tasks
    return tasks.memory_maze_9x9(global_observables=True, image_only_obs=False, camera_resolution=64, seed=seed)


def run(env, actions_fn, steps):
    ts = env.reset()
    obs, act, pos, dirs = [], [], [], []
    layout = ts.observation["maze_layout"].copy()
    for t in range(steps):
        a = actions_fn(t)
        o = ts.observation
        obs.append(o["image"]); pos.append(o["agent_pos"]); dirs.append(o["agent_dir"])
        act.append(np.eye(N_ACT, dtype=np.float32)[a])
        ts = env.step(a)
        if ts.last():
            break
    return dict(obs=np.array(obs), act=np.array(act), pos=np.array(pos, np.float32),
                dir=np.array(dirs, np.float32), layout=layout)


def job(args):
    kind, idx, steps, walk, turn, out = args
    rng = np.random.default_rng(idx + (0 if kind == "rollout" else 100000))
    env = make_env(int(idx) + (0 if kind == "rollout" else 100000))
    state = {"a": 1, "left": 0}

    def explore(t):
        if state["left"] == 0:
            state["a"], state["left"] = rng.choice(N_ACT, p=PROBS), rng.integers(1, 5)
        state["left"] -= 1
        return int(state["a"])

    if kind == "rollout":
        d = run(env, explore, steps)
    else:
        d = run(env, lambda t: explore(t) if t < walk else (0 if t < walk + 3 else 2), walk + 3 + turn)
        d["turn_start"] = walk + 3
    np.savez_compressed(os.path.join(out, f"{kind}_{idx:04d}.npz"), **d)
    return len(d["obs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n-turn", type=int, default=30)
    ap.add_argument("--walk", type=int, default=200)
    ap.add_argument("--turn", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/memmaze")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [("rollout", i, args.steps, 0, 0, args.out) for i in range(args.n)]
    jobs += [("turn", i, 0, args.walk, args.turn, args.out) for i in range(args.n_turn)]
    t0, total = time.time(), 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(job, jobs)):
            total += n
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes, {total} frames, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {total} frames in {time.time() - t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
