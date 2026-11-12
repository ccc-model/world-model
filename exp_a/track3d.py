"""實驗 A 第一階段（感知）：影片 → 每個追蹤物體每一幀的 3D 位置，存成 CSV。

YOLO11n + ByteTrack 給 2D 框和追蹤 ID，Depth Anything V2 metric 給深度，
取框中央區域的深度中位數當 Z，再用內參把框中心反投影成 (X, Y, Z)。

用法：python exp_a/track3d.py data/vtest.avi --hfov 60 --out out/vtest.csv
"""
import argparse, csv, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from ultralytics import YOLO

from wm.depth import MetricDepth, pick_device
from wm.geometry import intrinsics_from_hfov, backproject


def box_depth(depth, x1, y1, x2, y2, shrink=0.5):
    """框中央 shrink 比例區域的深度中位數（避開邊緣的背景像素）。"""
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * shrink, (y2 - y1) * shrink
    H, W = depth.shape
    a, b = int(max(cx - w / 2, 0)), int(min(cx + w / 2, W - 1)) + 1
    c, d = int(max(cy - h / 2, 0)), int(min(cy + h / 2, H - 1)) + 1
    return float(np.median(depth[c:d, a:b]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hfov", type=float, default=69.0, help="水平視角（度）；iPhone 主鏡頭約 69")
    ap.add_argument("--scene", default="outdoor", choices=["outdoor", "indoor"])
    ap.add_argument("--yolo", default="yolo11n.pt")
    ap.add_argument("--stride", type=int, default=1, help="每幾幀處理一次")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    out = args.out or os.path.join("out", os.path.splitext(os.path.basename(args.video))[0] + ".csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    device = pick_device()
    yolo = YOLO(args.yolo)
    depth_model = MetricDepth(args.scene, device=device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = intrinsics_from_hfov(W, H, args.hfov)
    print(f"{args.video}: {W}x{H} @ {fps:.1f}fps, fx={K[0,0]:.1f}, device={device}")

    f = open(out, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["frame", "t", "id", "cls", "name", "conf", "x1", "y1", "x2", "y2", "X", "Y", "Z"])
    i, n, t0 = -1, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % args.stride:
            continue
        r = yolo.track(frame, persist=True, tracker="bytetrack.yaml", conf=args.conf,
                       device=device, verbose=False)[0]
        if r.boxes.id is None:
            n += 1
            continue
        depth = depth_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for (x1, y1, x2, y2), tid, c, p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().tolist(),
                                               r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            Z = box_depth(depth, x1, y1, x2, y2)
            X, Y, Z = backproject((x1 + x2) / 2, (y1 + y2) / 2, Z, K)
            wr.writerow([i, f"{i / fps:.4f}", tid, c, r.names[c], f"{p:.3f}",
                         *(f"{v:.1f}" for v in (x1, y1, x2, y2)), f"{X:.3f}", f"{Y:.3f}", f"{Z:.3f}"])
        n += 1
        if n % 20 == 0:
            print(f"  frame {i}  {n / (time.time() - t0):.1f} fps")
        if args.max_frames and n >= args.max_frames:
            break
    f.close()
    # 把影片資訊寫在旁邊，第二階段要用
    with open(out + ".meta", "w") as m:
        m.write(f"video={os.path.abspath(args.video)}\nfps={fps}\nwidth={W}\nheight={H}\nhfov={args.hfov}\nstride={args.stride}\n")
    print(f"done: {n} frames in {time.time() - t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
