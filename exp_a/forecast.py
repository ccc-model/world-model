"""實驗 A 第二階段（動態）：讀 track3d 的 CSV，用 3D Kalman 濾波並預測未來軌跡。

輸出：
  - ADE/FDE：Kalman 等速模型 vs. 兩個基準（原地不動、用最後兩點算速度）
  - 鳥瞰圖 (X-Z 平面) PNG
  - 標註影片：框 + 深度 + 投影回畫面的未來軌跡

用法：python exp_a/forecast.py out/vtest.csv --horizon 1.0
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wm.geometry import intrinsics_from_hfov, project
from wm.kalman import CVKalman3D


def load(csv_path):
    meta = dict(l.strip().split("=", 1) for l in open(csv_path + ".meta") if "=" in l)
    rows = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    tracks = defaultdict(list)
    for r in np.atleast_1d(rows):
        tracks[int(r["id"])].append(r)
    return meta, tracks


def run(tracks, fps, stride, horizon, warmup, sigma_a):
    dt = stride / fps
    steps = max(1, round(horizon / dt))
    out = {}          # (frame, id) -> dict(filt, fc)
    err = {"static": [], "cv_raw": [], "kalman": []}
    for tid, rs in tracks.items():
        frames = [int(r["frame"]) for r in rs]
        P = np.array([[r["X"], r["Y"], r["Z"]] for r in rs], float)
        by_frame = dict(zip(frames, P))
        kf = None
        for j, (fr, p) in enumerate(zip(frames, P)):
            if kf is None:
                kf = CVKalman3D(p, dt, sigma_a=sigma_a)
            else:
                kf.predict((fr - frames[j - 1]) / fps)
                kf.update(p)
            fc = kf.forecast(steps)
            out[(fr, tid)] = dict(filt=kf.s[:3].copy(), fc=fc, cls=str(rs[j]["name"]))
            if j < warmup:
                continue
            # 評估：只用「未來每一步都有觀測」的時刻
            fut = [fr + k * stride for k in range(1, steps + 1)]
            if not all(f in by_frame for f in fut):
                continue
            gt = np.array([by_frame[f] for f in fut])
            prev = by_frame.get(fr - stride)
            preds = {"static": np.repeat(p[None], steps, 0), "kalman": fc}
            if prev is not None:
                v = (p - prev) / dt
                preds["cv_raw"] = np.array([p + v * dt * k for k in range(1, steps + 1)])
            for name, pr in preds.items():
                d = np.linalg.norm((pr - gt)[:, [0, 2]], axis=1)   # 鳥瞰平面誤差
                err[name].append((d.mean(), d[-1]))
    metrics = {k: {"ADE": float(np.mean([e[0] for e in v])), "FDE": float(np.mean([e[1] for e in v])), "n": len(v)}
               for k, v in err.items() if v}
    return out, metrics, steps


def plot_bev(tracks, out, path, every):
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab20")
    for i, (tid, rs) in enumerate(tracks.items()):
        if len(rs) < 10:
            continue
        c = cmap(i % 20)
        raw = np.array([[r["X"], r["Z"]] for r in rs])
        filt = np.array([out[(int(r["frame"]), tid)]["filt"][[0, 2]] for r in rs])
        ax.scatter(raw[:, 0], raw[:, 1], s=3, color=c, alpha=0.25)
        ax.plot(filt[:, 0], filt[:, 1], color=c, lw=1.5)
        for r in rs[::every]:
            fc = out[(int(r["frame"]), tid)]["fc"]
            ax.plot(fc[:, 0], fc[:, 2], "--", color=c, lw=0.8, alpha=0.7)
    ax.plot(0, 0, "k^", ms=10)
    ax.set_xlabel("X (m, right)"); ax.set_ylabel("Z (m, forward)")
    ax.set_title("Bird's-eye view: dots=raw, line=Kalman, dashed=forecast")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def render(meta, tracks, out, K, path):
    by_frame = defaultdict(list)
    for tid, rs in tracks.items():
        for r in rs:
            by_frame[int(r["frame"])].append((tid, r))
    cap = cv2.VideoCapture(meta["video"])
    fps, W, H = float(meta["fps"]), int(meta["width"]), int(meta["height"])
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps / int(meta["stride"]), (W, H))
    i = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i not in by_frame:
            continue
        for tid, r in by_frame[i]:
            color = tuple(int(c) for c in np.random.default_rng(tid).integers(60, 255, 3))
            x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"#{tid} {r['name']} {r['Z']:.1f}m", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            o = out[(i, tid)]
            pts = np.vstack([o["filt"], o["fc"]])
            pts = pts[pts[:, 2] > 0.3]
            if len(pts) > 1:
                uv = project(pts, K).astype(int)
                cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
                cv2.circle(img, tuple(uv[-1]), 4, color, -1)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--horizon", type=float, default=1.0, help="預測多少秒後")
    ap.add_argument("--warmup", type=int, default=5, help="前幾次觀測不計分")
    ap.add_argument("--sigma-a", type=float, default=1.0, help="加速度噪音 (m/s^2)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    meta, tracks = load(args.csv)
    fps, stride = float(meta["fps"]), int(meta["stride"])
    out, metrics, steps = run(tracks, fps, stride, args.horizon, args.warmup, args.sigma_a)

    base = os.path.splitext(args.csv)[0]
    print(f"tracks={len(tracks)}  horizon={args.horizon}s ({steps} steps)")
    print(f"{'model':8s} {'ADE(m)':>8s} {'FDE(m)':>8s} {'n':>6s}")
    for k, m in metrics.items():
        print(f"{k:8s} {m['ADE']:8.3f} {m['FDE']:8.3f} {m['n']:6d}")
    json.dump(metrics, open(base + "_metrics.json", "w"), indent=2)
    plot_bev(tracks, out, base + "_bev.png", every=max(1, steps))
    print("→", base + "_bev.png")
    if not args.no_video:
        K = intrinsics_from_hfov(int(meta["width"]), int(meta["height"]), float(meta["hfov"]))
        render(meta, tracks, out, K, base + "_forecast.mp4")
        print("→", base + "_forecast.mp4")


if __name__ == "__main__":
    main()
