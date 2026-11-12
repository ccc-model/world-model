"""讀取 collect.py 的 rollouts。最後 10% 回合當測試集。"""
import glob
import numpy as np


def rollout_files(root="data/carracing", split="train"):
    files = sorted(glob.glob(f"{root}/rollout_*.npz"))
    n = int(len(files) * 0.9)
    return files[:n] if split == "train" else files[n:]


def load_frames(files):
    return np.concatenate([np.load(f)["obs"] for f in files])
