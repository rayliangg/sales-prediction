# Amazon Product Naming vs Sales Predictor

這個專案會：
- 自動下載 Amazon Products Dataset（你提供的 KaggleHub 來源）
- 用產品名稱訓練「預測銷售量」模型
- 根據語意相似商品做正規化（同質產品比較），輸出 z-score 和百分位

## 1) 安裝

預設依賴為 **`faiss-gpu`**（需 NVIDIA + CUDA，例如 Colab T4、多數 Linux GPU 機）。

```bash
cd amazon-name-sales-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

若沒有 NVIDIA GPU（例如 Apple Silicon），請改裝 CPU 版 FAISS：

```bash
pip install -r requirements-cpu.txt
```

### 訓練用 GPU FAISS、推論用 CPU（建議部署方式）

- **訓練**（`train.py`）：安裝 `requirements.txt`（內含 **`faiss-gpu`**）。有 GPU 時會在 GPU 上建索引，再 **`index_gpu_to_cpu`** 後寫入 `models/name_sales_faiss.index`，檔案本身是 **CPU 索引**，不依賴 GPU。
- **推論 / Web**（`predict.py`、`web.py`）：只需 **`faiss.read_index` + `search`**，全程在 **CPU**。部署機若沒 GPU，請裝：

```bash
pip install -r requirements-inference.txt
```

本機裝 `faiss-gpu` 時推論同樣走 CPU，但若想減少依賴體積，推論環境用 `requirements-inference.txt` 即可。

## 2) 訓練模型

```bash
PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 250000
```

參數說明：
- `--sample-n`: 先抽樣幾筆訓練（0 代表全量資料，會慢很多）

## 3) 預測

```bash
PYTHONPATH=src python -m amazon_name_sales_predictor.predict \
  --model-path models/name_sales_model.joblib \
  --name "Wireless Bluetooth Earbuds Noise Cancelling"
```

輸出重點：
- `predicted_sales`: 預測銷售量
- `normalized_z_score`: 在相似商品中的標準分數（>0 表示高於同質平均）
- `normalized_percentile_in_similars`: 在相似商品中的百分位（例如 80 表示贏過 80% 相似品）
- `inferred_category`: 由相似商品推斷出的主要類別

## 4) Web 介面

安裝完依賴後，直接啟動：

```bash
PYTHONPATH=src streamlit run src/amazon_name_sales_predictor/web.py
```

介面功能：
- 輸入產品名稱即時預測
- 顯示同質平均、Z 分數、同質百分位
- 顯示 Top K 相似商品做對照（依側欄設定）

## 5) 你的任務對應方式

你目前的需求是 `input = 產品名稱`, `output = 可能銷售量`，並且要按同性質產品正規化。

本專案的做法：
1. 用名稱文字特徵（TF-IDF）預測銷售量
2. 用 FAISS（`IndexHNSWFlat` + L2 正規化後的內積，等同 cosine）找到語意相似商品（視為同性質）
3. 在這批相似商品中計算平均、標準差、百分位

這樣你可以知道：
- 絕對值：這個命名大概能賣多少
- 相對值：這個命名在同類商品中是強還弱

## 6) 注意

- 資料集欄位名稱如果變動，程式會自動嘗試匹配常見欄位。
- 如果你想提高精度，建議再加入 `price`、`review count`、`rating` 等欄位做多特徵模型。
- 若要改用 Colab 訓練，請看 `COLAB.md`。

