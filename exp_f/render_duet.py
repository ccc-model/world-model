"""實驗 F：把 duolando_follow.py 的輸出畫成 MuJoCo 雙木偶人影片，並計算真實 vs 生成的女方指標。

影片分欄：左「真實：男 + 真實女」，中「Duolando GPT：男 + 生成女」，若有 --rl 再加右「Duolando + RL」。
男方相同（藍），女方紅（真實）/ 橘（GPT）/ 綠（RL）。

用法：uv run python exp_f/render_duet.py --rl out/exp_f_rl     # 處理 out/exp_f/*.npz
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from exp_f.puppet import render_video, to_zup

BLUE, RED, ORANGE, GREEN = (0.2, 0.4, 0.9, 1), (0.85, 0.25, 0.3, 1), (0.95, 0.6, 0.1, 1), (0.2, 0.7, 0.35, 1)
L_WRIST, R_WRIST, FEET = 20, 21, [7, 8, 10, 11]


def stats(leader, follower, fps):
    """leader, follower: (T,55,3) z-up。"""
    dist = np.linalg.norm((follower[:, 0] - leader[:, 0])[:, :2], axis=1)                 # 兩人骨盆的水平距離
    hands = np.stack([np.linalg.norm(follower[:, a] - leader[:, b], axis=1)
                      for a in (L_WRIST, R_WRIST) for b in (L_WRIST, R_WRIST)], 1).min(1)    # 兩人最近的一對手腕
    speed = np.linalg.norm(np.diff(follower[:, :22], axis=0), axis=2).mean(1) * fps       # 身體關節平均速度
    torso = [0, 3, 6, 9, 12, 15]                                                          # 骨盆到頭的軀幹關節
    gap = np.linalg.norm(follower[:, torso, None] - leader[:, None, torso], axis=3).min((1, 2))
    feet = follower[:, FEET]
    low = feet[1:, :, 2] < 0.05 + feet[:, :, 2].min()                                      # 貼地的腳
    slide = (np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=2) * fps)[low]
    return {"partner_distance_m": float(dist.mean()), "partner_distance_std": float(dist.std()),
            "hand_contact_ratio": float((hands < 0.15).mean()), "hand_min_dist_m": float(np.median(hands)),
            "mean_speed_mps": float(speed.mean()), "frozen_ratio": float((speed < 0.1).mean()),
            "overlap_ratio": float((gap < 0.15).mean()),                                    # 軀幹互相穿過
            "foot_slide_mps": float(np.median(slide)) if slide.size else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/exp_f")
    ap.add_argument("--rl", default=None, help="RL 版的輸出資料夾（同檔名），有給就多一欄")
    ap.add_argument("--seconds", type=float, default=30, help="影片長度上限")
    args = ap.parse_args()
    report = {}
    files = sorted(glob.glob(f"{args.dir}/*.npz"))
    if not files:
        print(f"{args.dir}/ 裡沒有 .npz：請先執行 exp_f/duolando_follow.py 產生結果")
        return
    for f in files:
        d = np.load(f)
        fps = int(d["fps"])
        L, Fr, Fg = (to_zup(d[k]) for k in ("leader", "follower_real", "follower_gen"))
        name = os.path.basename(f)[:-4]
        gens = {"gpt": Fg}
        rl_file = args.rl and os.path.join(args.rl, os.path.basename(f))
        if rl_file and os.path.exists(rl_file):
            gens["rl"] = to_zup(np.load(rl_file)["follower_gen"])
        # 生成 vs 真實：根節點相對位置後的姿勢誤差（不同但合理的舞也算「誤差」，只供參考）
        rel = lambda P: P - P[:, :1]
        report[name] = {"real": stats(L, Fr, fps)}
        for k, G in gens.items():
            report[name][k] = stats(L, G, fps)
            report[name][k]["pose_error_m"] = float(np.linalg.norm(rel(G)[:, :22] - rel(Fr)[:, :22], axis=2).mean())
        n = int(args.seconds * fps)
        panels = [([L[:n], Fr[:n]], [BLUE, RED]), ([L[:n], Fg[:n]], [BLUE, ORANGE])]
        titles = ["real couple", "Duolando GPT"]
        if "rl" in gens:
            panels.append(([L[:n], gens["rl"][:n]], [BLUE, GREEN])); titles.append("Duolando + RL")
        render_video(panels, f"{args.dir}/{name}.mp4", fps=fps, titles=titles)
        print(name, json.dumps(report[name]), flush=True)
    json.dump(report, open(f"{args.dir}/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
