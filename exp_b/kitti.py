"""KITTI Tracking 讀取：labels + calib + oxts → 每個物體在世界座標的鳥瞰軌跡。

KITTI 的 3D 標註在「當下那一幀的相機座標」，相機裝在會動的車上，
所以要用 oxts (GPS/IMU) 算出每幀車子的位姿，把物體轉到固定的世界座標。

目錄結構（官網三個 zip 解壓到 data/kitti/）：
  data/kitti/training/label_02/0000.txt ...
  data/kitti/training/calib/0000.txt ...
  data/kitti/training/oxts/0000.txt ...
"""
import os
from collections import defaultdict
import numpy as np

CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram")


def _h(M):
    """3x3 或 3x4 → 4x4 齊次矩陣。"""
    T = np.eye(4)
    T[:M.shape[0], :M.shape[1]] = M
    return T


def load_calib(path):
    d = {}
    for line in open(path):
        parts = line.split()
        if parts:
            d[parts[0].rstrip(":")] = np.array(parts[1:], float)
    R_rect = _h(d["R_rect"].reshape(3, 3))
    Tr_velo_cam = _h(d["Tr_velo_cam"].reshape(3, 4))
    Tr_imu_velo = _h(d["Tr_imu_velo"].reshape(3, 4))
    # 矯正後相機座標 → IMU 座標
    return np.linalg.inv(Tr_imu_velo) @ np.linalg.inv(Tr_velo_cam) @ np.linalg.inv(R_rect)


def load_poses(path):
    """oxts → 每幀 IMU→世界 的 4x4 位姿（以第一幀為原點，KITTI devkit 的 Mercator 投影）。"""
    oxts = np.loadtxt(path, ndmin=2)
    er = 6378137.0
    scale = np.cos(oxts[0, 0] * np.pi / 180)
    poses = []
    for lat, lon, alt, roll, pitch, yaw in oxts[:, :6]:
        t = np.array([scale * lon * np.pi * er / 180,
                      scale * er * np.log(np.tan((90 + lat) * np.pi / 360)), alt])
        cx, sx, cy, sy, cz, sz = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rz @ Ry @ Rx, t
        poses.append(T)
    T0inv = np.linalg.inv(poses[0])
    return np.array([T0inv @ T for T in poses])


def load_sequence(root, seq, cam_noise=None, rng=None):
    """回傳 {track_id: dict(cls, frames, xy, cam)}。xy 是世界座標鳥瞰位置 (N,2)。

    cam_noise=(xy_rel, z_rel)：在相機座标加上和距離成正比的噪音，模擬實驗 A 的單眼深度。
    """
    s = f"{seq:04d}"
    cam2imu = load_calib(os.path.join(root, "calib", s + ".txt"))
    poses = load_poses(os.path.join(root, "oxts", s + ".txt"))
    tracks = defaultdict(lambda: dict(cls=None, frames=[], xy=[], cam=[]))
    for line in open(os.path.join(root, "label_02", s + ".txt")):
        f = line.split()
        frame, tid, cls = int(f[0]), int(f[1]), f[2]
        if tid < 0 or cls not in CLASSES:
            continue
        p = np.array(f[13:16], float)       # 相機座標，物體底面中心
        if cam_noise is not None:
            z = max(p[2], 1.0)
            p = p + rng.normal(0, 1, 3) * np.array([cam_noise[0] * z, 0, cam_noise[1] * z])
        w = poses[frame] @ cam2imu @ np.r_[p, 1.0]
        t = tracks[tid]
        t["cls"] = cls
        t["frames"].append(frame)
        t["xy"].append(w[:2])
        t["cam"].append(p)
    return {k: dict(cls=v["cls"], frames=np.array(v["frames"]), xy=np.array(v["xy"]), cam=np.array(v["cam"]))
            for k, v in tracks.items()}


def list_sequences(root):
    return sorted(int(n[:4]) for n in os.listdir(os.path.join(root, "label_02")) if n.endswith(".txt"))


def make_windows(tracks, n_obs, n_pred, seq=None):
    """把軌跡切成 (過去 n_obs 幀, 未來 n_pred 幀) 的連續視窗。"""
    out = []
    L = n_obs + n_pred
    for tid, t in tracks.items():
        fr, xy = t["frames"], t["xy"]
        for i in range(len(fr) - L + 1):
            if fr[i + L - 1] - fr[i] == L - 1:          # 中間沒有缺幀
                out.append(dict(seq=seq, tid=tid, cls=t["cls"], frame=int(fr[i + n_obs - 1]),
                                obs=xy[i:i + n_obs], fut=xy[i + n_obs:i + L]))
    return out
