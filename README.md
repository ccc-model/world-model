# world-model

世界模型實驗：從「看懂 3D」到「預測世界怎麼變」。目標硬體是 iMac M3（PyTorch `mps`）。

## 安裝

```bash
uv sync
```

## 實驗

| 實驗 | 內容 | 要訓練？ | 文件 |
|---|---|---|---|
| A | YOLO11n 追蹤 + Depth Anything V2 metric → 3D 軌跡 → Kalman 預測未來 | 否 | [doc/exp_a.md](doc/exp_a.md) |
| B1 | KITTI Tracking 3D 真值軌跡 → 小 GRU vs. 等速基準，含模擬單眼噪音 | 是（20 秒） | [doc/exp_b.md](doc/exp_b.md) |
| C | World Models (CarRacing)：VAE 潛在空間 + MDN-LSTM vs. MDN-Transformer 做夢 | 是（約 35 分鐘） | [doc/exp_c.md](doc/exp_c.md) |

## 結構

```
wm/        共用模組（相機幾何、深度模型、Kalman）
exp_a/     實驗 A：track3d.py（感知）、forecast.py（動態）
exp_b/     實驗 B1：kitti.py（讀取＋座標轉換）、train.py（GRU 與基準）
exp_c/     實驗 C：collect.py → train_vae.py → train_m.py → dream.py
doc/       實驗說明與結果
data/ out/ models/   資料、輸出、權重（不進 git）
```

## 成果一覽

| 實驗 A：單眼 3D 追蹤與預測 | 實驗 C：World Models 做夢 |
|---|---|
| ![A](doc/media/exp_a_forecast.gif) | ![C](doc/media/exp_c_dream.gif) |
