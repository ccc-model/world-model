"""實驗 E：跑現成的 diffusion 世界模型 DIAMOND（NeurIPS 2024）的 Atari 預訓練權重。

DIAMOND 的世界模型是一個 diffusion 模型：給過去 4 幀和動作，去噪 3 步生成下一幀（64x64）。
agent（actor-critic）完全在這個「夢」裡訓練。這支程式不開視窗，對每個遊戲做三件事：
  1. real：預訓練 agent 在真實遊戲裡玩 --steps 步
  2. open-loop dream：從同一個時間點開始，只給前 4 幀真實畫面，之後餵「真實的動作序列」讓世界模型自己生成
  3. agent-in-dream：同樣起點，但 agent 看著夢裡的畫面自己決定動作（真正的「在夢裡玩」）
輸出並排影片與開環誤差。

需用 DIAMOND 自己的環境執行（Python 3.10、torch 2.4.1）：
  external/diamond/.venv/bin/python exp_e/diamond_dream.py --games Breakout Pong Boxing MsPacman
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAMOND = ROOT / "external" / "diamond"
sys.path.insert(0, str(DIAMOND / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical

from agent import Agent
from envs import make_atari_env
from models.diffusion import DiffusionSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def download(f):
    return Path(hf_hub_download(repo_id="eloialonso/diamond", filename=f))


def to_img(x):
    """(3,H,W) [-1,1] → (H,W,3) uint8 RGB"""
    return ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def run_game(name, args, device):
    with initialize_config_dir(version_base="1.3", config_dir=str(DIAMOND / "config")):
        cfg = compose(config_name="trainer")
    ckpt = download(f"atari_100k/models/{name}.pt")
    cfg.agent = OmegaConf.load(download("atari_100k/config/agent/default.yaml"))
    cfg.env = OmegaConf.load(download("atari_100k/config/env/atari.yaml"))
    cfg.env.test.id = f"{name}NoFrameskip-v4"
    env = make_atari_env(num_envs=1, device=device, **cfg.env.test)
    agent = Agent(instantiate(cfg.agent, num_actions=env.num_actions)).to(device).eval()
    agent.load(ckpt)
    ac = agent.actor_critic
    sampler = DiffusionSampler(agent.denoiser, instantiate(cfg.world_model_env.diffusion_sampler))
    n_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
    n_params = sum(p.numel() for p in agent.denoiser.parameters())

    # 1) 真實遊戲：agent 玩 warmup + steps 步
    obs, _ = env.reset(seed=[args.seed])
    hx = torch.zeros(1, ac.lstm_dim, device=device); cx = torch.zeros_like(hx)
    O, A, H = [], [], []
    for t in range(args.warmup + args.steps):
        logits, _, (hx, cx) = ac.predict_act_value(obs, (hx, cx))
        a = Categorical(logits=logits).sample()
        O.append(obs[0]); A.append(a[0]); H.append((hx.clone(), cx.clone()))
        obs, rew, end, trunc, _ = env.step(a)
        if (end + trunc).any():
            break
    env.close()
    O, A = torch.stack(O), torch.stack(A)
    t0 = args.warmup
    T = min(args.steps, len(O) - t0)

    def dream(policy):
        ob = O[None, t0 - n_cond:t0].clone()
        ac_buf = A[None, t0 - n_cond:t0].clone()
        h = H[t0 - 2]                      # agent 的記憶：看完 t0-1 之前的真實畫面
        out, acts = [], []
        for k in range(T):
            if policy == "real":
                a = A[t0 - 1 + k]
            else:
                logits, _, h = ac.predict_act_value(ob[:, -1], h)
                a = Categorical(logits=logits).sample()[0]
            ac_buf[:, -1] = a
            nxt, _ = sampler.sample(ob, ac_buf)
            out.append(nxt[0]); acts.append(int(a))
            ob = ob.roll(-1, 1); ac_buf = ac_buf.roll(-1, 1); ob[:, -1] = nxt
        return torch.stack(out), acts

    ts = time.time()
    open_loop, _ = dream("real")
    if device.type == "mps":
        torch.mps.synchronize()
    fps = T / (time.time() - ts)
    in_dream, _ = dream("agent")
    real = O[t0:t0 + T]
    mse = ((open_loop - real) ** 2 / 4).mean((1, 2, 3)).cpu().numpy()          # 換算成 [0,1] 像素的 MSE
    copy = ((real[:1].expand_as(real) - real) ** 2 / 4).mean((1, 2, 3)).cpu().numpy()
    return dict(real=real, open_loop=open_loop, in_dream=in_dream, mse=mse, copy=copy,
                fps=fps, steps=T, params=n_params, actions=env.num_actions)


def write_video(path, frames_by_col, labels, title, fps=15, S=256):
    T = len(frames_by_col[0])
    W = S * len(frames_by_col)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, S + 24))
    for t in range(T):
        bar = np.zeros((24, W, 3), np.uint8)
        for i, lab in enumerate(labels):
            text = f"{title} t={t + 1} | {lab}" if i == 0 else lab
            cv2.putText(bar, text, (6 + i * S, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        row = np.concatenate([cv2.resize(to_img(c[t]), (S, S), interpolation=cv2.INTER_NEAREST) for c in frames_by_col], 1)
        vw.write(cv2.cvtColor(np.concatenate([bar, row], 0), cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["Breakout", "Pong", "Boxing", "MsPacman"])
    ap.add_argument("--device", default="cpu", help="模型只有 4M 參數，CPU 比 mps 略快")
    ap.add_argument("--warmup", type=int, default=60, help="真實遊戲先玩幾步再開始做夢")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = ROOT / "out"; out.mkdir(exist_ok=True)
    summary = {}
    for g in args.games:
        r = run_game(g, args, device)
        write_video(out / f"exp_e_{g}.mp4", [r["real"], r["open_loop"], r["in_dream"]],
                    ["real game", "dream (same actions)", "agent plays in dream"], g)
        m = r["mse"]
        summary[g] = dict(params=int(r["params"]), actions=int(r["actions"]), steps=int(r["steps"]), dream_fps=round(r["fps"], 1),
                          mse={k: float(m[k - 1]) for k in (1, 10, 50, 100, 200) if k <= len(m)},
                          copy_first={k: float(r["copy"][k - 1]) for k in (1, 10, 50, 100, 200) if k <= len(m)},
                          curve=m.tolist())
        print(f"{g:10s} denoiser {r['params'] / 1e6:.1f}M params, {r['steps']} steps, "
              f"dream {r['fps']:.1f} frames/s on {args.device}, MSE@1/10/50/100: "
              + " ".join(f"{summary[g]['mse'].get(k, float('nan')):.4f}" for k in (1, 10, 50, 100)), flush=True)
    json.dump(summary, open(out / "exp_e_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
