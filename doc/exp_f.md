# 實驗 F：雙人舞的「跟舞」模型 Duolando（DD100 Jive）

前面的世界模型預測的是「畫面」或「物體位置」。這裡換成**人體動作**：
給真實的男方（leader）動作和音樂，讓模型生成女方（follower）怎麼跳，再放進 MuJoCo 讓兩個木偶人一起跳。
女方的模型本質上就是一個互動世界模型：$\text{女方}_{t+1} = f(\text{女方}_{\le t},\ \text{男方}_{\le t+k},\ \text{音樂})$。

- 模型：[Duolando](https://github.com/lisiyao21/Duolando)（Li et al.，ICLR 2024），南洋理工大學 S-Lab，授權為 NTU S-Lab License（僅限非商業用途）。
- 資料：論文自建的 **DD100** 雙人舞資料集，10 個舞種，含 **Jive（吉魯巴）**。
- 只用公開的預訓練權重與測試集，**不訓練、不微調**。

```mermaid
flowchart LR
  M["音樂特徵"] --> G
  L["男方動作<br/>（真實，SMPL-X 55 關節）"] -->|"VQ-VAE 編碼"| LT["男方 token"]
  LT --> G["follower GPT<br/>206M 參數"]
  F0["女方第 1 個 token"] --> G
  G -->|"自迴歸生成"| FT["女方 token<br/>（姿勢 + 相對位移）"]
  FT -->|"VQ-VAE 解碼"| F["女方動作"]
  F --> R["MuJoCo 雙木偶人"]
  L --> R
```

## 模型

Duolando 分三個部分，都是公開權重：

| 部分 | 作用 | 參數 | 權重大小 |
|---|---|---|---|
| 動作 VQ-VAE | 把上半身、下半身、雙手的姿勢壓成離散 token | 116.5M | 470 MB |
| 位移 VQ-VAE | 把「女方相對男方的位置」壓成 token | 17.3M | 70 MB |
| follower GPT | 看音樂、男方 token、自己過去的 token，預測女方下一個 token | 206M | 825 MB |
| follower GPT + RL | 同上，再用離線強化學習微調，獎勵懲罰腳滑與互相穿過 | 206M | 825 MB |

生成時只給女方**第一個 token**（約 8 幀）當起點，之後整段都由模型自己往下滾。男方動作則完整給定，所以模型看得到男方接下來要做什麼。

## 安裝

Duolando 用 Python 3.10，放在獨立環境；`external/` 不進 git。

```bash
git clone https://github.com/lisiyao21/Duolando.git external/duolando
```

```bash
cd external/duolando && uv venv --python 3.10 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install torch==2.4.1 "numpy<2" scipy easydict pyyaml tqdm matplotlib opencv-python smplx
```

從 Duolando README 的 Google Drive 連結下載兩個檔案，解壓在 `external/duolando/` 下：

| 檔案 | 內容 | 大小 | 下載時間（約 0.5 MB/s） |
|---|---|---|---|
| `data__lazy.zip` | DD100 前處理後的動作與音樂特徵 → `data/` | 5.1 GB（解開 6.4 GB） | 2 小時 17 分 |
| `experiments.zip` | 四個權重 → `experiments/` | 2.0 GB | 46 分 |

```bash
cd external/duolando && uvx gdown 1MpoytmnSGbiVLSOL0QKvGz4dmW_8TrH6 && uvx gdown 1DT9fzaz7M2ls7dqS1jwJdQc3Cy78_9Oy
```

```bash
cd external/duolando && unzip -q data__lazy.zip && unzip -q experiments.zip
```

### 讓它在 Mac 上跑

原始程式只支援 CUDA。[`exp_f/duolando_follow.py`](../exp_f/duolando_follow.py) 不改 Duolando 的程式碼，而是：

- 把 `torch.Tensor.cuda`、`nn.Module.cuda` 換成「搬到 CPU」，因為模型裡有很多寫死的 `.cuda()`。
- `pytorch3d` 在 Mac 上難裝，而 Duolando 只 import 它、沒有真的用到，所以放一個替身 [`exp_f/_shims/pytorch3d`](../exp_f/_shims/pytorch3d)。
- 推論流程照抄 Duolando 的 `eval()`，只挑指定舞種。

## 執行

```bash
external/duolando/.venv/bin/python exp_f/duolando_follow.py --genre Jive
```

```bash
external/duolando/.venv/bin/python exp_f/duolando_follow.py --genre Jive --config configs/rl_final_debug_reward3_random_5mem_lr3e-5.yaml --out out/exp_f_rl
```

```bash
uv run python exp_f/render_duet.py --rl out/exp_f_rl
```

前兩行分別用 GPT 版與 RL 版生成女方，輸出 `out/exp_f*/<片段>.npz`（男方、真實女方、生成女方各 $(T, 55, 3)$ 關節位置）。
第三行用 [`exp_f/puppet.py`](../exp_f/puppet.py) 在 MuJoCo 畫膠囊木偶人、錄影，並計算指標寫進 `out/exp_f/metrics.json`。
木偶人只做運動學播放：每一幀把關節位置畫進場景，不跑物理模擬。

## 時間（iMac M3 8GB，CPU）

DD100 測試集共 34 段，其中 Jive 4 段，每段 48–64 秒。

| 步驟 | 時間 |
|---|---|
| 生成一段（GPT 或 RL） | 84–176 秒，約為舞蹈長度的 2 倍 |
| 4 段 × 2 版本 | 約 15 分鐘 |

## 結果

### 影片

左：真實的一對（藍男、紅女）；中：Duolando GPT 生成的女方（橘）；右：GPT + RL（綠）。
三欄的藍色男方完全相同，都是真實動作捕捉資料。

![Jive 跟舞](media/exp_f_jive.gif)

完整 30 秒（H.264）：[media/exp_f_Jive_010_001.mp4](media/exp_f_Jive_010_001.mp4)

第 20 秒的畫面：

![第 20 秒](media/exp_f_frame.png)

真實女方站在男方側邊、手搭著男方；GPT 版的女方幾乎疊在男方前面；RL 版拉開到旁邊，比較像真的雙人舞站位。

### 指標

4 段 Jive 的平均。「手相觸」以兩人最近一對手腕距離小於 15 cm 計；「穿過」以兩人軀幹關節（骨盆到頭）最近距離小於 15 cm 計；
「腳滑」是腳貼地（離最低點 5 cm 內）時的水平速度。

| 指標 | 真實女方 | GPT | GPT + RL |
|---|---|---|---|
| 兩人骨盆水平距離 | 1.10 m | 0.99 m | 1.19 m |
| 距離的標準差 | 0.29 m | 0.38 m | 0.45 m |
| 手相觸的幀數比例 | 8.2% | 2.9% | 1.9% |
| 最近手腕距離（中位數） | 0.47 m | 0.54 m | 0.67 m |
| 軀幹互相穿過的幀數比例 | 0% | 1.0% | 0.5% |
| 身體關節平均速度 | 0.88 m/s | 1.20 m/s | 1.01 m/s |
| 腳滑速度（中位數） | **0.09 m/s** | 1.20 m/s | 0.93 m/s |
| 與真實女方的姿勢差（相對骨盆） | — | 0.18 m | 0.18 m |

以 Jive_010_001 進一步看骨盆和腳：

| | 真實 | GPT | GPT + RL |
|---|---|---|---|
| 骨盆水平速度 | 0.49 m/s | 0.83 m/s | 0.77 m/s |
| 腳離地高度（中位數） | 4 cm | 18 cm | 18 cm |

## 觀察

1. **看得出是在跳 Jive**。從影片看，生成的女方會踏步、轉身、擺手，動作風格像 Jive；是否踩在拍點上，這裡沒有量。和真實女方的姿勢差約 18 cm，但這不代表錯：同一段男方可以有很多種合理的女方跳法。
2. **最弱的是「雙人互動」**。真實舞者有 8% 的時間手接觸，生成的只有 2–3%。這個模型以 token 為單位預測姿勢與相對位移，沒有「手要握住對方」的硬性約束，接觸只能靠統計湊出來。
3. **腳沒有踩住地面**。生成女方的腳中位數離地 18 cm，而且骨盆移動得比真人快。姿勢 token 和位移 token 是分開解碼再相加的，沒有東西強迫「著地的腳不能動」，所以看起來像在滑冰。
4. **RL 微調確實有改善，但有限**。穿過的比例減半（1.0% → 0.5%），腳滑降約 2 成，第 20 秒的畫面裡站位拉開到側邊，比較像真的雙人舞。但它是用拉開距離換來的：手接觸反而更少。這正是實驗 E 和 [世界模型與強化學習](rl_vs_world_model.md) 提到的「策略鑽獎勵漏洞」：獎勵懲罰穿模，最省事的解就是離遠一點。
5. **對照前面的實驗**：
   - 和 C、D 一樣，它是在離散或壓縮空間裡自迴歸往前滾。不同的是，它每一步都看得到真實的男方動作和音樂，所以不會塌縮成「平均動作」。這對應到 E 的 Breakout：有外部輸入一直拉著，夢才不會走偏。
   - 它不懂物理。重力、地面、接觸都只是統計，和 C、D 的 VAE 世界模型一樣。

## 延伸：10 個舞種

DD100 測試集的 34 段全部跑一次，GPT 與 RL 各一次：

```bash
external/duolando/.venv/bin/python exp_f/duolando_follow.py --genre all
```

```bash
external/duolando/.venv/bin/python exp_f/duolando_follow.py --genre all --config configs/rl_final_debug_reward3_random_5mem_lr3e-5.yaml --out out/exp_f_rl
```

```bash
uv run python exp_f/render_duet.py --rl out/exp_f_rl --videos-per-genre 1
```

生成 30 段新片段（每版本約 16 分鐘的舞）兩個版本共花 2 小時 22 分，其中有一段時間和 G1 錄影搶 CPU。

各舞種平均，每格是「真實 / GPT / RL」：

| 舞種 | 段數 | 兩人距離 (m) | 手相觸比例 | 軀幹穿過比例 | 腳滑 (m/s) |
|---|---|---|---|---|---|
| Waltz 華爾滋 | 3 | 0.25 / 0.26 / 0.26 | **0.99** / 0.15 / 0.14 | 0 / 0.05 / 0.04 | 0.05 / 0.90 / 0.61 |
| Tango 探戈 | 3 | 0.27 / 0.27 / 0.26 | **0.99** / 0.12 / 0.12 | 0 / 0.09 / 0.10 | 0.05 / 0.86 / 0.68 |
| Foxtrot 狐步 | 3 | 0.26 / 0.28 / 0.29 | **0.91** / 0.18 / 0.19 | 0 / 0.02 / 0.03 | 0.04 / 0.97 / 0.65 |
| Quickstep 快步 | 3 | 0.30 / 0.28 / 0.28 | **0.95** / 0.13 / 0.13 | 0 / 0.05 / 0.06 | 0.12 / 1.12 / 0.91 |
| PasoDable 鬥牛舞 | 4 | 0.78 / 0.72 / 0.74 | 0.35 / 0.05 / 0.03 | 0 / 0.01 / 0.00 | 0.04 / 0.83 / 0.82 |
| Qiaqiaqia 恰恰 | 4 | 0.89 / 0.80 / 0.92 | 0.19 / 0.03 / 0.02 | 0 / 0.01 / 0.01 | 0.05 / 0.74 / 0.49 |
| Samba 森巴 | 3 | 1.11 / 0.89 / 1.11 | 0.11 / 0.02 / 0.02 | 0 / 0.01 / 0.00 | 0.06 / 1.12 / 0.56 |
| Jive 吉魯巴 | 4 | 1.10 / 0.99 / 1.19 | 0.08 / 0.03 / 0.02 | 0 / 0.01 / 0.00 | 0.09 / 1.20 / 0.93 |
| Lumba 倫巴 | 3 | 1.07 / 0.95 / 1.41 | 0.05 / 0.02 / 0.01 | 0 / 0.01 / 0.00 | 0.03 / 0.68 / 0.48 |
| Ballet 芭蕾 | 4 | 1.10 / 0.58 / 0.58 | 0.06 / 0.03 / 0.03 | 0 / 0.00 / 0.00 | 0.02 / 0.58 / 0.58 |

Tango（每種舞各錄一段，在 `out/exp_f/`）：

![Tango](media/exp_f_tango.gif)

完整 30 秒（H.264）：[media/exp_f_Tango_005_022.mp4](media/exp_f_Tango_005_022.mp4)

- **閉式舞姿學到了「站位」，沒學到「握手」**。華爾滋、探戈、狐步、快步這四種標準舞，真實舞者 91–99% 的時間牽著手，兩人距離約 0.26 m。模型生成的女方把距離抓得幾乎一模一樣，手相觸卻只剩 12–19%。它知道「要貼著跳」，但沒有學到「手要握住」這種精確的接觸約束。
- **貼得越近，穿模越多**。閉式舞姿的軀幹穿過比例 2–10%，探戈最嚴重；分開跳的拉丁舞只有 0–1%。
- **RL 對腳滑普遍有效，對握手和穿模幾乎沒效**。腳滑在 8 種舞降了約 2–5 成（芭蕾和鬥牛舞沒變），但手相觸和穿過比例幾乎不變。這和 RL 的獎勵設計一致：獎勵主要懲罰腳滑和穿過，閉式舞姿下兩人本來就得貼近，沒有「離遠一點」的空間可鑽。
- **芭蕾是例外**：真實兩人距離 1.10 m，生成的只有 0.58 m，RL 也沒改。芭蕾的雙人動作和其他舞差很多，模型可能把它當成一般的貼身舞來跳。
- **所有舞種的腳滑都遠大於真人**：GPT 版是真人的 9–29 倍，RL 版仍有 7 倍以上，這是姿勢和位移分開預測的根本限制（見上面的觀察 3）。

## 延伸：套到 Unitree G1 機器人（運動學重定向）

把同樣的動作套到兩台 [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)（29 個關節、身高約 1.3 m）。
火柴人沒有體積，看不太出穿模；換成有外殼的機器人，兩台的碰撞網格一重疊就能被 MuJoCo 偵測到。
這一步只做運動學（每幀直接設定關節角），還沒有物理：機器人不會跌倒，但關節極限和身體體積都是真的。

```bash
cd external && git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git && cd mujoco_menagerie && git sparse-checkout set unitree_g1
```

```bash
uv run python exp_f/g1_retarget.py out/exp_f/Jive_010_002.npz --rl out/exp_f_rl
```

[`exp_f/g1_retarget.py`](../exp_f/g1_retarget.py) 的做法：

1. **骨長換算**：G1 比人矮，比例也不同。沿著人體骨架，用「人的骨頭方向 × G1 的骨頭長度」算出 G1 16 個關鍵點（骨盆、髖、膝、踝、腳尖、肩、肘、腕、頭）的目標位置；根節點的位移依骨盆高度比例縮放，兩人共用同一個比例，相對位置才不會變形。
2. **逆運動學**：每一幀用阻尼最小平方法解兩台 G1 的 `qpos`（含浮動基座共 2 × 36 維），

$$
\Delta q = J^\top (J J^\top + \lambda I)^{-1} r ,
$$

   $r$ 是加權後的關鍵點誤差，$J$ 是 `mj_jacSite` 算出的 Jacobian。以上一幀的解為起點迭代 8 次，關節角夾在 G1 的關節極限內。
3. **碰撞檢查**：`mj_forward` 之後，看兩台機器人之間的接觸，記錄最深的穿透深度。

一段 30 秒（900 幀）三欄約 2 分鐘，每欄 35–45 秒。

![G1 版](media/exp_f_g1.gif)

完整 30 秒（H.264）：[media/exp_f_g1_Jive_010_002.mp4](media/exp_f_g1_Jive_010_002.mp4)

第 12 秒：真實女方（紅）貼在男方側邊；GPT 版（橘）站在男方前面、手搭在一起；RL 版（綠）離得很遠，沒有互動。

![G1 第 12 秒](media/exp_f_g1_frame.png)

4 段 Jive 前 30 秒的平均：

| 指標 | 真實女方 | GPT | GPT + RL |
|---|---|---|---|
| 兩台機器人網格穿透 > 1 cm 的幀數比例 | **1.3%** | 11.1% | 8.9% |
| 穿透深度第 95 百分位 | 0.1 cm | 4.7 cm | 3.2 cm |
| 女方 IK 誤差（G1 跟不上目標的程度） | 2.5 cm | 5.5 cm | 5.1 cm |
| 女方腳踝離地高度（中位數） | 1.5 cm | 4.5 cm | 4.3 cm |

各段差異很大：

| 片段 | GPT 穿透比例 | RL 穿透比例 |
|---|---|---|
| Jive_005_001 | 6.8% | 7.9% |
| Jive_005_002 | 13.8% | 0% |
| Jive_010_001 | 4.7% | **18.0%** |
| Jive_010_002 | 19.0% | 9.7% |

- **有體積之後，穿模明顯多了**。火柴人只用軀幹中心線判斷，GPT 版只有 1% 的幀穿過；換成 G1 的外殼網格，變成 11%。真實的一對也有 1.3%，因為 G1 的身形和真人不同，貼身時會擦到。
- **RL 平均有改善，但不穩定**。四段裡兩段變好、一段持平、Jive_010_001 反而從 4.7% 變成 18%。只有 4 段，不足以下結論。
- **生成的動作比較難重現**。G1 跟真實女方動作的誤差是 2.5 cm，跟生成動作卻有 5 cm。生成的姿勢有時超出 G1 的關節極限，或骨頭方向前後不一致，表示它比真人動作「不自然」。
- **還不是物理**。這裡機器人不會跌倒。真正要考驗動作可不可行，需要用物理模擬加上追蹤控制器（例如 DeepMimic、PHC）讓 G1 自己用力矩跟著跳。

## 授權

Duolando 的程式碼與權重採 NTU S-Lab License 1.0，只能用於非商業用途；DD100 資料也以研究用途公開。本實驗只做教學與研究上的推論與視覺化。

## 可以接著做

- **把男方換成生成的**：先用單人的音樂生舞模型（例如同作者的 Bailando；EDGE 需要 Jukebox 特徵，8GB 跑不動）生成男方，再讓 Duolando 跟，就是完全「夢出來」的雙人舞。
- **加上物理**：在 G1 重定向的基礎上，用追蹤控制器讓 G1 在物理模擬中跟著跳。這樣腳滑、穿模會直接變成跌倒，是比這裡的指標更嚴格的檢查。現成的追蹤策略多半要 NVIDIA Isaac，在 Mac 上要自己在 MuJoCo 訓練，需要數天。
- **用 SMPL-X 網格取代火柴人**：`data/motion/smplx` 已經有參數，可以畫出有體積的人，比較容易看出穿模。
