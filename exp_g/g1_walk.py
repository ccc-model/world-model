"""實驗 G：載入 Unitree 官方預訓練的 G1 行走策略，在 MuJoCo 物理模擬裡走路、錄影，並做推擠測試。

策略與模型來自 unitree_rl_gym（BSD-3，external/unitree_rl_gym，不進 git）：
  git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_gym.git external/unitree_rl_gym
  cd external/unitree_rl_gym && git sparse-checkout set deploy/deploy_mujoco deploy/pre_train/g1 resources/robots/g1_description

控制迴圈照抄官方 deploy/deploy_mujoco/deploy_mujoco.py：
  物理 500 Hz；策略 50 Hz 輸出 12 個腿部關節的目標角，PD 換成力矩。
差別：不開視窗（離屏錄影）、速度指令依時間表變化、可以在骨盆上施加側向推力。

用法：uv run python exp_g/g1_walk.py              # 錄影 + 推擠測試
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "glfw")
import cv2
import mujoco
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import yaml

from exp_f.puppet import SCENE_XML

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URG = os.path.join(ROOT, "external", "unitree_rl_gym")
CFG = yaml.safe_load(open(os.path.join(URG, "deploy", "deploy_mujoco", "configs", "g1.yaml")))

# 速度指令時間表：(開始秒, [前進 m/s, 側移 m/s, 轉向 rad/s], 說明)
SCHEDULE = [(0, [0.5, 0, 0], "forward 0.5 m/s"), (6, [0.5, 0, 0.6], "turn left"), (11, [0.0, 0.3, 0], "side step"),
            (15, [0.0, 0, 0], "stand (stepping in place)"), (19, [0.8, 0, 0], "forward 0.8 m/s"), (25, [0.5, 0, 0], "push test")]
PUSHES = [(27.0, 150.0), (30.0, -150.0), (33.0, 300.0)]   # (秒, 側向推力 N)，各推 0.2 秒；最後一次會推倒


def gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


def build():
    scene = mujoco.MjSpec.from_string(SCENE_XML)
    scene.worldbody.geoms[0].size = [30, 30, 0.1]        # 走得比木偶人遠，地板放大到 60 m
    scene.material("grid").texrepeat = [40, 40]
    g1 = mujoco.MjSpec.from_file(os.path.join(URG, "resources", "robots", "g1_description", "g1_12dof.xml"))
    g1.meshdir = os.path.join(URG, "resources", "robots", "g1_description", "meshes")
    g1.option.timestep = scene.option.timestep
    scene.worldbody.add_frame().attach_body(g1.body("pelvis"), "g1_", "")
    m = scene.compile()
    m.opt.timestep = CFG["simulation_dt"]
    return m


class Walker:
    def __init__(self):
        self.m = build()
        self.d = mujoco.MjData(self.m)
        self.policy = torch.jit.load(os.path.join(URG, "deploy", "pre_train", "g1", "motion.pt"))
        self.kp, self.kd = np.array(CFG["kps"]), np.array(CFG["kds"])
        self.q0 = np.array(CFG["default_angles"])
        self.n = CFG["num_actions"]
        self.pelvis = self.m.body("g1_pelvis").id
        self.d.qpos[7:] = self.q0
        mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(self.n)
        self.target = self.q0.copy()
        self.counter = 0

    def obs(self, cmd):
        d, c = self.d, CFG
        t = self.counter * c["simulation_dt"]
        phase = t % 0.8 / 0.8
        return np.concatenate([d.qvel[3:6] * c["ang_vel_scale"], gravity_orientation(d.qpos[3:7]),
                               np.array(cmd) * np.array(c["cmd_scale"]),
                               (d.qpos[7:] - self.q0) * c["dof_pos_scale"], d.qvel[6:] * c["dof_vel_scale"],
                               self.action, [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]]).astype(np.float32)

    def step(self, cmd, push=0.0):
        """前進一個物理步（2 ms）。push: 骨盆受到的側向力（N，世界座標 y）。"""
        d = self.d
        d.ctrl[:] = (self.target - d.qpos[7:]) * self.kp - d.qvel[6:] * self.kd
        d.xfrc_applied[self.pelvis, :3] = [0, push, 0]
        mujoco.mj_step(self.m, d)
        self.counter += 1
        if self.counter % CFG["control_decimation"] == 0:
            with torch.no_grad():
                self.action = self.policy(torch.from_numpy(self.obs(cmd))[None]).numpy().squeeze()
            self.target = self.action * CFG["action_scale"] + self.q0

    @property
    def t(self):
        return self.counter * CFG["simulation_dt"]

    def fallen(self):
        return self.d.qpos[2] < 0.45                      # 骨盆站立高度約 0.78 m


def body_vel(w):
    """骨盆在自身朝向座標下的 (前進, 側移) 速度與轉向角速度。"""
    q = w.d.qpos[3:7]
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    vx, vy = w.d.qvel[0], w.d.qvel[1]
    return np.array([np.cos(yaw) * vx + np.sin(yaw) * vy, -np.sin(yaw) * vx + np.cos(yaw) * vy, w.d.qvel[5]])


def record(out, seconds=37, fps=30, size=(960, 540)):
    w = Walker()
    rend = mujoco.Renderer(w.m, size[1], size[0])
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    log, next_frame, fell_at = [], 0.0, None
    while w.t < seconds:
        i = max(k for k, s in enumerate(SCHEDULE) if w.t >= s[0])
        cmd, label = SCHEDULE[i][1], SCHEDULE[i][2]
        push = next((f for s, f in PUSHES if s <= w.t < s + 0.2), 0.0)
        w.step(cmd, push)
        if fell_at is None and w.fallen():
            fell_at = w.t
        if w.counter % CFG["control_decimation"] == 0:
            log.append((w.t, i, *cmd, *body_vel(w)))
        if w.t >= next_frame:
            next_frame += 1 / fps
            cam.lookat[:] = [w.d.qpos[0], w.d.qpos[1], 0.6]
            cam.distance, cam.azimuth, cam.elevation = 2.8, 120, -15
            rend.update_scene(w.d, cam)
            img = rend.render().copy()
            v = body_vel(w)
            txt = [f"t={w.t:5.1f}s  {label}", f"cmd  vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} yaw={cmd[2]:+.2f}",
                   f"real vx={v[0]:+.2f} vy={v[1]:+.2f} yaw={v[2]:+.2f}"]
            if push:
                txt.append(f"PUSH {push:+.0f} N")
            for k, s in enumerate(txt):
                cv2.putText(img, s, (12, 28 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 30, 30) if "PUSH" in s else (20, 20, 20), 2 if "PUSH" in s else 1, cv2.LINE_AA)
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    log = np.array(log)
    stats = {}
    for i, (t0, cmd, label) in enumerate(SCHEDULE):
        seg = log[(log[:, 1] == i) & (log[:, 0] >= t0 + 2) & (log[:, 0] < 33)]          # 每段前 2 秒是加速過程，不算
        if len(seg):
            stats[label] = {"cmd": cmd, "real_mean": seg[:, 5:8].mean(0).round(3).tolist()}
    stats["fell_at_s"] = fell_at
    stats["distance_m"] = float(np.linalg.norm(w.d.qpos[:2]))
    return stats


def push_test(forces, cmd=(0.5, 0, 0), duration=0.2, at=4.0, after=4.0):
    """直走中，在骨盆上推一次側向力，看接下來 4 秒會不會倒。"""
    res = {}
    for f in forces:
        w = Walker()
        fell = False
        while w.t < at + after:
            w.step(cmd, f if at <= w.t < at + duration else 0.0)
            fell |= w.fallen()
        res[f] = bool(fell)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/exp_g")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = record(f"{args.out}/g1_walk.mp4")
    print(json.dumps(stats, indent=1), flush=True)
    forces = [100, 150, 160, 170, 180, 190, 200, 300]
    pt = push_test(forces)
    print("push 0.2 s, fell? ", pt, flush=True)
    stats["push_test_fell"] = {str(k): v for k, v in pt.items()}
    json.dump(stats, open(f"{args.out}/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
