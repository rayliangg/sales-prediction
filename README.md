# Amazon Product Naming vs Sales Predictor

這個專案會：
- 自動下載 Amazon Products Dataset（你提供的 KaggleHub 來源）
- 用產品名稱訓練「預測銷售量」模型
- 根據語意相似商品做正規化（同質產品比較），輸出 z-score 和百分位

## 1) 安裝

```bash
cd amazon-name-sales-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
2. 用 HNSW（Hierarchical Navigable Small World）找到語意相似商品（視為同性質）
3. 在這批相似商品中計算平均、標準差、百分位

這樣你可以知道：
- 絕對值：這個命名大概能賣多少
- 相對值：這個命名在同類商品中是強還弱

## 6) 注意

- 資料集欄位名稱如果變動，程式會自動嘗試匹配常見欄位。
- 如果你想提高精度，建議再加入 `price`、`review count`、`rating` 等欄位做多特徵模型。
- 若要改用 Colab 訓練，請看 `COLAB.md`。

