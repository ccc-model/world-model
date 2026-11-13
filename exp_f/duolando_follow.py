"""實驗 F：用 Duolando（ICLR 2024）的預訓練模型做「跟舞」。

給 DD100 測試集裡的真實男方動作（leader）與音樂，讓 Duolando 的 follower GPT 生成女方動作，
並和真實女方比較。流程沿用 Duolando 的 MCTall.eval()，差別：
  - 不改它的程式碼：把 .cuda() 導向 CPU（它的模型裡有很多寫死的 .cuda()）
  - 只跑指定舞種（預設 Jive）
  - 輸出 (T, 55, 3) 的關節位置：leader、真實 follower、生成的 follower

需用 Duolando 自己的環境執行（Python 3.10、torch 2.4.1）：
  external/duolando/.venv/bin/python exp_f/duolando_follow.py --genre Jive
"""
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUO = ROOT / "external" / "duolando"

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

DEV = torch.device("cpu")
torch.Tensor.cuda = lambda self, *a, **k: self.to(DEV)
nn.Module.cuda = lambda self, *a, **k: self.to(DEV)

sys.path.insert(0, str(DUO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_shims"))   # pytorch3d 替身
_CWD = os.getcwd()
os.chdir(DUO)                                    # 設定檔裡的路徑都是相對於 repo
import models                                     # noqa: E402
from datasets.dd100lf_all2 import DD100lfAll      # noqa: E402


def to_abs(pose):
    """Duolando 的表示（根節點絕對位置 + 其他關節相對根節點、手指放大 10 倍）→ (T, 55, 3) 絕對位置。"""
    pose = pose.copy()
    lt, rt = pose[:, 60:63], pose[:, 63:66]
    pose[:, 75:120] = pose[:, 75:120] * 0.1 + np.tile(lt, (1, 15))
    pose[:, 120:165] = pose[:, 120:165] * 0.1 + np.tile(rt, (1, 15))
    root = pose[:, :3]
    pose = pose + np.tile(root, (1, 55))
    pose[:, :3] = root
    return pose.reshape(len(pose), 55, 3)


def load(module, path):
    sd = torch.load(path, map_location=DEV)["model"]
    module.load_state_dict(sd, strict=True)
    return module.eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/follower_gpt_beta0.9_final.yaml")
    ap.add_argument("--ckpt", default=None, help="follower GPT 權重；預設 experiments/<expname>/ckpt/epoch_<ckpt_epoch>.pt")
    ap.add_argument("--genre", default="Jive")
    ap.add_argument("--out", default=str(ROOT / "out" / "exp_f"))
    args = ap.parse_args()
    args.out = os.path.join(_CWD, args.out)      # 相對路徑以啟動時的資料夾為準
    cfg = EasyDict(yaml.safe_load(open(args.config)))
    os.makedirs(args.out, exist_ok=True)

    vqvae = nn.DataParallel(getattr(models, cfg.structure.name)(cfg.structure).float()).to(DEV)
    gpt = nn.DataParallel(getattr(models, cfg.structure_generate.name)(cfg.structure_generate).float()).to(DEV)
    tvq = nn.DataParallel(getattr(models, cfg.structure_transl_vqvae.name)(cfg.structure_transl_vqvae).float()).to(DEV)
    load(vqvae, cfg.vqvae_weight)
    load(tvq, cfg.transl_vqvae_weight)
    ckpt = args.ckpt or f"experiments/{cfg.expname}/ckpt/epoch_{cfg.testing.ckpt_epoch}.pt"
    load(gpt, ckpt)
    n_params = sum(p.numel() for p in gpt.parameters())
    print(f"follower GPT {n_params / 1e6:.1f}M params from {ckpt}", flush=True)

    d = cfg.data.test
    ds = DD100lfAll(d.music_root, d.data_root, split=d.split, interval=None, dtype=d.dtype, move=d.move, music_dance_rate=1)
    genres = sorted({n.split("_")[0] for n in ds.names})
    idx = [i for i, n in enumerate(ds.names) if n.split("_")[0].lower() == args.genre.lower()]
    print(f"test set: {len(ds)} clips, genres {genres}; {args.genre}: {len(idx)} clips", flush=True)

    for i in idx:
        b = {k: torch.as_tensor(np.asarray(v))[None] if k != "fname" else v for k, v in ds[i].items()}
        name = b["fname"]
        music, pl, pf = b["music"].float(), b["pos3dl"].float().clone(), b["pos3df"].float().clone()
        real_f, real_l = pf[0].numpy().copy(), pl[0].numpy().copy()

        # 以下照抄 Duolando 的 eval()：follower 相對 leader 的位移另外用 transl VQ-VAE 編碼
        lftransl = (pf[:, :, :3] - pl[:, :, :3]).clone() * 20.0
        transll = pl[:, :, :3].clone()
        transll = transll - transll[:, :1, :3].clone()
        pl[:, :, :3] = 0
        pf[:, :, :3] = 0
        ql = vqvae.module.encode(pl)
        qf = vqvae.module.encode(pf[:, :cfg.ds_rate])           # 只給女方第一個 token 當起點
        qt = tvq.module.encode(lftransl)
        y = tuple(ql[k][0].clone() for k in range(len(qf)))
        x = tuple(qf[k][0][:, :1].clone() for k in range(len(qf)))
        t0 = time.time()
        zs, transl_z = gpt.module.sample(x + (qt[0][:, :1],), cond=(music[:, cfg.music_motion_rate:],) + y)
        pose, _, _ = vqvae.module.decode(zs)
        lf = tvq.module.decode(transl_z)
        pose[:, :, :3] = transll[:, :lf.size(1)] + lf / 20.0
        gen = pose[0].numpy()

        T = len(gen)
        leader = real_l[:T].copy(); leader[:, :3] = leader[:, :3] - real_l[:1, :3]
        follower = real_f[:T].copy(); follower[:, :3] = follower[:, :3] - real_l[:1, :3]
        np.savez_compressed(os.path.join(args.out, f"{name}.npz"), leader=to_abs(leader),
                            follower_real=to_abs(follower), follower_gen=to_abs(gen), fps=30)
        print(f"  {name}: {T} frames ({T / 30:.1f}s), generated in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
