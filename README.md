# CollabLoc

本專案以 [hloc](https://github.com/cvg/Hierarchical-Localization) 為基礎修改。
根目錄的 `README_hloc.md` 是 hloc 原本的說明;本檔記錄 CollabLoc 自己的部分。

外部程式碼的來源請見 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 交接說明:如何使用這份程式

完整的交接文件在 HackMD(含詳細參數、截圖與 Google Drive 連結),本節是整理後的流程摘要:

- 建模(Model Construction):https://hackmd.io/JceIab7qRaKnvgWgD4Y43Q
- 定位與評估(Localization):https://hackmd.io/vKDh_zrtSI2s_9SjZA6vaw

整體流程:**拍攝 → 抽幀 → 建模 → 啟動 Server → Android Client 定位 → 取得 ground truth → 評估**。

### 0. 建立環境

```bash
conda env create -f collabloc.yml
conda activate collabloc
```
`requirements.txt` 已棄用。離線評估另有環境檔 `arcore_eval_offline/collabloc-eval.yml`。

### 1. 拍攝與抽幀

- 緩慢、平順地移動鏡頭,避免快速轉向;光線要足夠,避開鏡子、玻璃反光與移動物體;轉角處放慢以拍到更多細節。
- 窄走廊可左右掃動鏡頭前進,回程再掃一次;長走廊可用大半徑螺旋方式連續拍攝。
- 用 `extract_video_frame.py`(Python 3 + `opencv-python`)或 VLC 從影片抽幀。主要參數:`gap`(每隔幾幀存一張,越小越多張)、`input_dir`、`output_dir`(需先建立)、`video_idx_range`、`video_pre`、`output_pre`、`show_im`。

### 2. 建模

在 `datasets/<project_name>/` 下放:
- `db_images/`:所有資料庫影像
- `ref_sfm_model/`:`cameras.bin`、`images.bin`、`points3D.bin`

做法有兩種:

1. **Colmap SfM 後再由 Server 三角化(建議)**
   - 用 Colmap 的 automatic reconstructor 建稀疏模型(quality high、single camera、不做 dense)。結果在 `<workspace>/sparse/0/`,耗時可能數小時。
   - 三角化:
     ```bash
     python3 -m hloc.pipelines.VBgps.triangulation --name <project_name>
     ```
     結果在 `outputs/<project_name>/`,定位用的模型是 `sfm_superpoint+NN/`。
2. **直接由 Server 建模**(較簡單但較容易失敗,因為 SuperPoint 特徵比 Colmap 的 SIFT 少)
   ```bash
   python3 -m hloc.pipelines.VBgps.Sfm --name <project_name>
   ```

接著抽取定位用特徵(兩種方式都要做):
```bash
python3 -m hloc.pipelines.VBgps.extract_features --name <project_name>
# 選用:--model RAFT/models/raft-small-old.pth --small
# GPU 記憶體不足時,拿掉選用參數
```
`extract_db_global` 已棄用,請用 `extract_features`。RAFT 權重的位置見下方「被忽略的檔案」。

Dense 重建(`image_undistorter` → `patch_match_stereo` → `stereo_fusion`)為選用,定位不需要。

### 3. 設定並啟動 Server

1. 取得最新穩定版 Server 程式(HackMD 內有 Google Drive 連結)。
2. 修改 [hloc/pipelines/VBgps/collabloc.py](hloc/pipelines/VBgps/collabloc.py) 中的 `addr`,使其等於 Server 的網路位址(目前程式碼內是 `('140.113.195.250', 9999)`,換機器必須改)。
3. 調整參數:`Cmin`(預設 0.2)、`Creset`(預設 0.7)、`αd`(預設 0.005)、`αa`(預設 0.0003)。離線評估時在 `client_simulator.py` 設定;線上(實機)時在 Android 的 `HelloArActivity.java` 設定,且數值為離線的 10 倍。
4. 用 `arcore_eval_offline/src/eval/calc_scale2.py` 由兩張照片及其 AR 位姿與 ground truth 位姿計算 scale(舊版 `calc_scale.py` 已棄用)。
5. 把 `<project_name>.yaml` 放進 `config/`,啟動 `collabloc` pipeline(`hloc/pipelines/VBgps/collabloc.py`,實際參數以 HackMD 與程式內 argparse 為準)。等看到 "server is ready" 再啟動 Client。

### 4. Android Client

- 用 Android Studio 開啟最新版 Client 程式,設定 Server 網路資訊,必要時設定 Colmap 格式的觀察位姿。
- 在支援 ARCore 的手機上執行,**先啟動 Server**。點螢幕開始或暫停記錄。
- 每次啟動會以 `yyyy-MM-dd_HH-mm-ss` 建立 log 資料夾,主要檔案為 `log.txt`、`sampled_imgs_log.txt`、`sampled_imgs/`。

### 5. Ground truth

- **Colmap incremental SfM:** 複製工作區,加入手機取樣影像,執行 [incremental_sfm.sh](incremental_sfm.sh),會產生 `gt_poses.txt`。**不要在原始模型上跑**,它會修改 `database.db`。
- **現場量測:** 記錄時實際量測位姿,寫入 `onsite_poses.txt`。

### 6. 評估

- **離線**(`arcore_eval_offline/src/eval/`):用 `client_simulator.py` 把 log 重放給 Server,再用 `track_vis.py`、`sim_multi_clients.sh`、`overview_logs2.py`、`onsite_measurement.py` 分析。
- **線上**(`arcore_eval`):用 `evaluation2.py`、`position_evaluation.py`、`visual.py`、`onsite_measurement.py` 分析實機記錄的融合位姿。

### 本倉庫沒有的內容

下列項目在 HackMD 中提到,但**不在這個資料夾裡**,需要從交接資料(Google Drive / GitHub)另外取得:
- 最新版 Server 程式、Android Client 原始碼
- `arcore_eval/`(線上評估)及其腳本
- `extract_video_frame.py`、舊版 `calc_scale.py`
- 拍攝用的原始影片、照片與已建好的模型

### 已知問題

- `hloc.pipelines.VBgps.quick_test` 已棄用,目前沒有替代方案。
- 本節是 HackMD 內容的整理;若兩者不一致,以 HackMD 與程式碼為準。

## 被 .gitignore 忽略、執行前需要自行放回的檔案

下列內容不在版本管理內。在新環境 clone 之後,必須先把它們放回指定位置才能執行。

### 模型權重

| 路徑 | 大小 | 用途 |
|---|---|---|
| `third_party/SuperGluePretrainedNetwork/models/weights/model_epoch_51.pth` | 約 2.0GB | 自訓練的 SuperGlue3D 完整模型物件。由 `third_party/SuperGluePretrainedNetwork/load_sg3d.py` 以 `torch.load` 載入,`hloc/matchers/superglue3d.py` 呼叫,對應 `match_features.confs['superglue3d']`(`hloc/pipelines/Aachen/pipeline_2d3d.py` 有使用)。 |
| `third_party/SuperGluePretrainedNetwork/models/weights/model_statedict_epoch_51.pth` | 約 48MB | 同一個模型只存權重(state_dict)的版本。目前程式沒有載入它,`match_features.py` 設定中寫了它的名稱,但 `load_sg_model` 並未使用該設定。 |

RAFT 的權重放在 `RAFT/models/`(例如 `raft-small-old.pth`、`raft-small.pth`),`extract_features` 的 `--model` 參數會用到。`RAFT/.gitignore` 內有 `models` 一行,所以這個資料夾不會被版控,同樣需要自行放回。

其餘 SuperGlue 權重(`superglue_indoor.pth`、`superglue_outdoor.pth`、`superpoint_v1.pth`)沒有被忽略,隨專案一起版控。

### 資料與輸出

| 路徑 | 說明 |
|---|---|
| `datasets/` | 各場景的影像與 COLMAP 資料(含 `database.db`、`medical/` 等)。僅根目錄的 `datasets/` 被忽略,`third_party/r2d2/datasets/` 的程式碼有版控。 |
| `config/` | 各場景的設定 yaml(約 20 個小檔)。已被忽略,換機器時需要自行複製。 |
| `outputs/` | 特徵、匹配、重建結果,可由 pipeline 重新產生。 |
| `time_info/` | 各次實驗的耗時紀錄,由 `hloc/pipelines/VBgps/collabloc_spa.py` 寫入 `time_info/collabloc_time_*.txt`。重跑不一定得到相同數值,如需保留請另外備份。 |
| `collabloc_time.txt` | 耗時彙整檔(約 17MB)。 |

## 備註

- `third_party/` 底下的 `d2net`、`r2d2`、`SuperGluePretrainedNetwork`、`deep-image-retrieval` 原本是 hloc 的 git submodule,現在以一般檔案納入本專案。其中 `SuperGluePretrainedNetwork` 有本地修改(`load_sg3d.py` 與 `models/superglue.py` 的 3D 版本)。
- `RAFT/` 同樣已移除原本的 `.git`,來源與版本見 THIRD_PARTY.md。
