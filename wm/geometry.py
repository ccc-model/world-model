"""相機幾何：內參與反投影 / 投影。"""
import math
import numpy as np


def intrinsics_from_hfov(width, height, hfov_deg):
    """沒有標定資料時，用水平視角估計內參 K（假設主點在影像中心、方形像素）。"""
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=float)


def backproject(u, v, Z, K):
    """像素 (u,v) + 深度 Z(公尺) → 相機座標 (X,Y,Z)。X 向右、Y 向下、Z 向前。"""
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    return np.array([X, Y, Z])


def project(P, K):
    """相機座標 (N,3) → 像素 (N,2)。"""
    P = np.atleast_2d(P)
    uv = (K @ P.T).T
    return uv[:, :2] / uv[:, 2:3]
