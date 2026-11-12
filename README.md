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

## 結構

```
wm/        共用模組（相機幾何、深度模型、Kalman）
exp_a/     實驗 A：track3d.py（感知）、forecast.py（動態）
doc/       實驗說明與結果
data/ out/ models/   資料、輸出、權重（不進 git）
```
