"""在 MuJoCo 裡畫「木偶人」：輸入每人每幀的 SMPL-X 55 個關節 3D 位置，畫成膠囊火柴人並錄影。

只做運動學播放（不算物理）：每幀把骨架畫進 MuJoCo 的場景（user geoms），用 mujoco.Renderer 離屏渲染。
macOS 需要 MUJOCO_GL=glfw。
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import cv2
import mujoco
import numpy as np

# SMPL-X 55 關節的父節點：0–21 身體、22 下巴、23/24 眼睛、25–39 左手、40–54 右手
_BODY = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15]
_HAND = lambda wrist, s: [wrist if i % 3 == 0 else s + i - 1 for i in range(15)]   # 5 指 × 3 節
PARENTS = np.array(_BODY + _HAND(20, 25) + _HAND(21, 40))
HAND_JOINTS = set(range(25, 55))

SCENE_XML = """
<mujoco>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".75 .82 .92" rgb2=".95 .96 .98" width="200" height="200"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".85 .85 .85" rgb2=".75 .75 .78" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom type="plane" size="6 6 0.1" material="grid"/>
  </worldbody>
</mujoco>
"""


def to_zup(pos):
    """pos: (T,55,3)。自動找出「上」是哪個軸（頭 - 腳踝），轉成 MuJoCo 的 z 朝上，並讓腳貼地。"""
    up_vec = (pos[:, 15] - pos[:, [7, 8]].mean(1)).mean(0)
    ax = int(np.argmax(np.abs(up_vec)))
    sign = np.sign(up_vec[ax])
    if ax == 2:
        out = pos * np.array([1, 1, sign])
    elif ax == 1:      # y-up → z-up：(x, y, z) → (x, -z, y)
        out = np.stack([pos[..., 0], -pos[..., 2] * sign, pos[..., 1] * sign], -1)
    else:              # x-up
        out = np.stack([pos[..., 1], pos[..., 2], pos[..., 0] * sign], -1)
    feet = out[:, [7, 8, 10, 11], 2]
    out[..., 2] -= np.percentile(feet.min(1), 1) - 0.03   # 腳趾關節離地約 3 cm
    return out


def _add_capsule(scene, a, b, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
    scene.ngeom += 1


def _add_sphere(scene, p, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), p, np.eye(3).ravel(), rgba)
    scene.ngeom += 1


class PuppetRenderer:
    def __init__(self, width=640, height=480):
        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)          # 算出地板等靜態物件的姿態
        self.renderer = mujoco.Renderer(self.model, height, width, max_geom=2000)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def frame(self, people, colors, lookat, distance=4.5, azimuth=90, elevation=-15):
        """people: list of (55,3) z-up 關節位置。回傳 RGB 影像。"""
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.renderer.update_scene(self.data, self.cam)
        sc = self.renderer.scene
        for P, rgba in zip(people, colors):
            for j, p in enumerate(PARENTS):
                if p < 0:
                    continue
                r = 0.008 if j in HAND_JOINTS else 0.035
                _add_capsule(sc, P[p], P[j], r, rgba)
            _add_sphere(sc, P[15] + np.array([0, 0, 0.08]), 0.1, rgba)      # 頭
        return self.renderer.render()


def render_video(panels, out_path, fps=30, size=(640, 480), titles=None, azimuth=90):
    """panels: list of (list of (T,55,3) z-up 位置, list of rgba)。每個 panel 一欄，並排錄影。"""
    T = min(len(P) for people, _ in panels for P in people)
    allpos = np.concatenate([P[:T] for people, _ in panels for P in people], 0)
    R = PuppetRenderer(*size)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[0] * len(panels), size[1]))
    for t in range(T):
        tiles = []
        for i, (people, colors) in enumerate(panels):
            center = np.mean([P[t, 0] for P in people], 0)              # 跟著兩人中心
            lookat = np.array([center[0], center[1], 0.9])
            img = R.frame([P[t] for P in people], colors, lookat, azimuth=azimuth).copy()
            if titles:
                cv2.putText(img, f"{titles[i]}  t={t / fps:5.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
            tiles.append(img)
        vw.write(cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
    vw.release()
    return T
