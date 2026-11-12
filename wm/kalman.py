"""3D 等速 (constant-velocity) Kalman 濾波器：最簡單的「物件中心世界模型」。

狀態 s = (X, Y, Z, Vx, Vy, Vz)，轉移 s_{t+1} = F s_t + 噪音。
觀測只有位置；單眼深度的誤差大約和距離成正比，所以 R 隨 Z 放大。
"""
import numpy as np


class CVKalman3D:
    def __init__(self, p0, dt, sigma_a=1.0, xy_rel=0.02, z_rel=0.10):
        self.dt, self.sigma_a, self.xy_rel, self.z_rel = dt, sigma_a, xy_rel, z_rel
        self.s = np.r_[p0, 0.0, 0.0, 0.0]
        z = max(p0[2], 1.0)
        self.P = np.diag([(xy_rel * z) ** 2] * 2 + [(z_rel * z) ** 2] + [4.0] * 3)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def _FQ(self, dt):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # 白噪音加速度模型
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt ** 4 / 4
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt ** 3 / 2
        Q[3:, 3:] = np.eye(3) * q * dt ** 2
        return F, Q

    def predict(self, dt=None):
        F, Q = self._FQ(dt or self.dt)
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q

    def update(self, p):
        z = max(p[2], 1.0)
        R = np.diag([(self.xy_rel * z) ** 2] * 2 + [(self.z_rel * z) ** 2])
        y = p - self.H @ self.s
        S = self.H @ self.P @ self.H.T + R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + Kg @ y
        self.P = (np.eye(6) - Kg @ self.H) @ self.P

    def forecast(self, steps, dt=None):
        """不加觀測地往前滾 steps 步，回傳 (steps,3) 未來位置。這就是「想像未來」。"""
        dt = dt or self.dt
        pos, vel = self.s[:3], self.s[3:]
        return np.array([pos + vel * dt * k for k in range(1, steps + 1)])
