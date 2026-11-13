"""實驗 F-3a：把雙人舞的關節位置重定向到兩台 Unitree G1（運動學，不跑物理）。

做法（和 GMR 的想法相同，但自己寫，不加依賴）：
  1. 骨長換算：沿人體骨架，用「人的骨頭方向 × G1 的骨頭長度」算出 G1 各關鍵點的目標位置；
     根節點的位移依身高比例縮放。
  2. 逆運動學：每幀用阻尼最小平方法（damped least squares）求兩台 G1 的 qpos，
     讓 G1 的關鍵點（site）貼近目標，關節角夾在 G1 的關節極限內；上一幀的解當起點。
  3. 用 MuJoCo 的碰撞偵測檢查兩台機器人的網格有沒有互相穿過，順便量腳離地高度。

G1 模型來自 mujoco_menagerie（external/mujoco_menagerie/unitree_g1，不進 git）：
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git external/mujoco_menagerie
  cd external/mujoco_menagerie && git sparse-checkout set unitree_g1

用法：uv run python exp_f/g1_retarget.py out/exp_f/Jive_010_001.npz --rl out/exp_f_rl
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "glfw")
import cv2
import mujoco
import numpy as np

from exp_f.puppet import SCENE_XML, to_zup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G1_XML = os.path.join(ROOT, "external", "mujoco_menagerie", "unitree_g1", "g1.xml")

# 關鍵點：(名稱, SMPL-X 關節, G1 body, site 在 body 座標的位置)
KEYS = [
    ("pelvis", 0, "pelvis", (0, 0, 0)),
    ("lhip", 1, "left_hip_roll_link", (0, 0, 0)), ("rhip", 2, "right_hip_roll_link", (0, 0, 0)),
    ("lknee", 4, "left_knee_link", (0, 0, 0)), ("rknee", 5, "right_knee_link", (0, 0, 0)),
    ("lankle", 7, "left_ankle_roll_link", (0, 0, 0)), ("rankle", 8, "right_ankle_roll_link", (0, 0, 0)),
    ("ltoe", 10, "left_ankle_roll_link", (0.12, 0, -0.03)), ("rtoe", 11, "right_ankle_roll_link", (0.12, 0, -0.03)),
    ("head", 15, "torso_link", (0, 0, 0.42)),
    ("lsho", 16, "left_shoulder_roll_link", (0, 0, 0)), ("rsho", 17, "right_shoulder_roll_link", (0, 0, 0)),
    ("lelb", 18, "left_elbow_link", (0, 0, 0)), ("relb", 19, "right_elbow_link", (0, 0, 0)),
    ("lwri", 20, "left_wrist_yaw_link", (0, 0, 0)), ("rwri", 21, "right_wrist_yaw_link", (0, 0, 0)),
]
NAMES = [k[0] for k in KEYS]
# 骨架樹（parent 名稱）；"neck" 是兩肩中點的虛擬點，不做 IK
TREE = {"lhip": "pelvis", "rhip": "pelvis", "lknee": "lhip", "rknee": "rhip", "lankle": "lknee", "rankle": "rknee",
        "ltoe": "lankle", "rtoe": "rankle", "neck": "pelvis", "head": "neck", "lsho": "neck", "rsho": "neck",
        "lelb": "lsho", "relb": "rsho", "lwri": "lelb", "rwri": "relb"}
WEIGHT = {"pelvis": 3.0, "lankle": 2.0, "rankle": 2.0, "ltoe": 1.5, "rtoe": 1.5}


def build(colors):
    """一個場景放 len(colors) 台 G1，名稱前綴 r0_、r1_…"""
    scene = mujoco.MjSpec.from_string(SCENE_XML)
    for i, rgba in enumerate(colors):
        g1 = mujoco.MjSpec.from_file(G1_XML)
        g1.meshdir = os.path.join(os.path.dirname(G1_XML), "assets")
        g1.option.integrator = scene.option.integrator    # 避免 attach 衝突警告
        for name, _, body, pos in KEYS:
            g1.body(body).add_site(name=f"kp_{name}", pos=pos, size=[0.01, 0, 0], group=5)
        for m in g1.materials:
            if m.name == "metal":
                m.rgba = rgba
        for k in list(g1.keys):                          # stand keyframe 只對單台有效
            g1.delete(k)
        frame = scene.worldbody.add_frame()
        frame.attach_body(g1.body("pelvis"), f"r{i}_", "")
    return scene.compile()


class Retarget:
    def __init__(self, colors):
        self.m = build(colors)
        self.d = mujoco.MjData(self.m)
        self.n = len(colors)
        self.sites = [[self.m.site(f"r{i}_kp_{n}").id for n in NAMES] for i in range(self.n)]
        self.free = [self.m.joint(f"r{i}_floating_base_joint") for i in range(self.n)]
        # 站姿：G1 的 stand keyframe（手肘彎曲），每台機器人各一份
        stand = np.array([0, 0, 0.79, 1, 0, 0, 0] + [0] * 15 + [0.2, 0.2, 0, 1.28, 0, 0, 0, 0.2, -0.2, 0, 1.28, 0, 0, 0])
        for i in range(self.n):
            a = self.free[i].qposadr[0]
            self.d.qpos[a:a + 36] = stand
        mujoco.mj_kinematics(self.m, self.d)
        P = self.d.site_xpos[self.sites[0]].copy()
        kp = dict(zip(NAMES, P))
        kp["neck"] = (kp["lsho"] + kp["rsho"]) / 2
        self.bone = {c: np.linalg.norm(kp[c] - kp[p]) for c, p in TREE.items()}
        self.pelvis_h = kp["pelvis"][2]
        lim = self.m.jnt_limited.astype(bool)
        self.hinge = [j for j in range(self.m.njnt) if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.lim = [(self.m.jnt_qposadr[j], *self.m.jnt_range[j]) for j in self.hinge if lim[j]]
        self.w = np.repeat([WEIGHT.get(n, 1.0) for n in NAMES], 3)

    def targets(self, H, human_pelvis_h):
        """H: (55,3) z-up 人體關節 → G1 各關鍵點目標位置 dict。"""
        h = {n: H[j] for n, j, _, _ in KEYS}
        h["neck"] = (h["lsho"] + h["rsho"]) / 2
        s = self.pelvis_h / human_pelvis_h
        t = {"pelvis": H[0] * s}
        for c in ["lhip", "rhip", "lknee", "rknee", "lankle", "rankle", "ltoe", "rtoe",
                  "neck", "head", "lsho", "rsho", "lelb", "relb", "lwri", "rwri"]:
            p = TREE[c]
            v = h[c] - h[p]
            t[c] = t[p] + v / (np.linalg.norm(v) + 1e-9) * self.bone[c]
        return np.array([t[n] for n in NAMES])

    def solve(self, targets, iters=12, lam=0.05):
        """targets: 每台機器人 (K,3)。回傳每台的關鍵點殘差 (公尺)。"""
        m, d = self.m, self.d
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, res = [], []
            for i in range(self.n):
                for s in self.sites[i]:
                    mujoco.mj_jacSite(m, d, jac, None, s)
                    rows.append(jac.copy())
                res.append((targets[i] - d.site_xpos[self.sites[i]]).ravel())
            J = np.concatenate(rows) * np.tile(self.w, self.n)[:, None]
            r = np.concatenate(res) * np.tile(self.w, self.n)
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(r)), r)
            mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
            for a, lo, hi in self.lim:
                d.qpos[a] = np.clip(d.qpos[a], lo, hi)
        mujoco.mj_kinematics(m, d)
        return [np.linalg.norm(targets[i] - d.site_xpos[self.sites[i]], axis=1).mean() for i in range(self.n)]

    def contacts(self):
        """兩台機器人之間的穿透深度（公尺，0 = 沒有碰到）。"""
        mujoco.mj_forward(self.m, self.d)
        deepest = 0.0
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.body_rootid[self.m.geom_bodyid[c.geom1]], self.m.body_rootid[self.m.geom_bodyid[c.geom2]]
            if b1 != b2 and b1 != 0 and b2 != 0:
                deepest = max(deepest, -c.dist)
        return deepest


def retarget_pair(leader, follower, color_f, renderer=None, fps=30, size=(640, 480), title=""):
    """回傳 (影格 list, 指標 dict)。leader/follower: (T,55,3) z-up。"""
    R = Retarget([(0.2, 0.4, 0.9, 1), color_f])
    hp = [np.mean([np.percentile(P[:, 0, 2], 90) for P in (leader, follower)])] * 2   # 兩人共用一個縮放，相對位置才一致
    T = min(len(leader), len(follower))
    frames, err, pen, feet = [], [], [], []
    rend = mujoco.Renderer(R.m, size[1], size[0], max_geom=4000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    for t in range(T):
        tg = [R.targets(leader[t], hp[0]), R.targets(follower[t], hp[1])]
        err.append(R.solve(tg, iters=30 if t == 0 else 8))
        pen.append(R.contacts())
        f_feet = [R.d.site_xpos[R.sites[1][NAMES.index(n)]][2] for n in ("lankle", "rankle")]
        feet.append(min(f_feet))
        center = (R.d.qpos[R.free[0].qposadr[0]:][:3] + R.d.qpos[R.free[1].qposadr[0]:][:3]) / 2
        cam.lookat[:] = [center[0], center[1], 0.7]
        cam.distance, cam.azimuth, cam.elevation = 3.2, 90, -12
        rend.update_scene(R.d, cam)
        img = rend.render().copy()
        cv2.putText(img, f"{title}  t={t / fps:5.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        frames.append(img)
    err, pen, feet = np.array(err), np.array(pen), np.array(feet)
    stats = {"ik_error_leader_m": float(err[:, 0].mean()), "ik_error_follower_m": float(err[:, 1].mean()),
             "robot_contact_ratio": float((pen > 0.01).mean()), "robot_penetration_p95_m": float(np.percentile(pen, 95)),
             "follower_ankle_height_median_m": float(np.median(feet))}
    return frames, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--rl", default=None, help="RL 版輸出資料夾，有給就多一欄")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out", default="out/exp_f_g1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.npz)
    fps = int(d["fps"]); n = int(args.seconds * fps)
    L = to_zup(d["leader"])[:n]
    cols = [("real couple", to_zup(d["follower_real"])[:n], (0.85, 0.25, 0.3, 1)),
            ("Duolando GPT", to_zup(d["follower_gen"])[:n], (0.95, 0.6, 0.1, 1))]
    if args.rl:
        cols.append(("Duolando + RL", to_zup(np.load(os.path.join(args.rl, os.path.basename(args.npz)))["follower_gen"])[:n],
                     (0.2, 0.7, 0.35, 1)))
    name = os.path.basename(args.npz)[:-4]
    report, panels = {}, []
    for title, F, c in cols:
        t0 = time.time()
        frames, st = retarget_pair(L, F, c, fps=fps, title=title)
        st["seconds"] = round(time.time() - t0, 1)
        report[title] = st; panels.append(frames)
        print(name, title, json.dumps(st), flush=True)
    T = min(len(p) for p in panels)
    h, w = panels[0][0].shape[:2]
    vw = cv2.VideoWriter(f"{args.out}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * len(panels), h))
    for t in range(T):
        vw.write(cv2.cvtColor(np.concatenate([p[t] for p in panels], 1), cv2.COLOR_RGB2BGR))
    vw.release()
    json.dump(report, open(f"{args.out}/{name}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
