# Amazon Product Naming vs Sales Predictor

從 Amazon 商品資料訓練：**產品名稱 → 預測銷量**，並以 FAISS 找相似品做 z-score / 百分位正規化。

## 安裝

```bash
cd amazon-name-sales-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**GPU 建索引（CUDA 12 / Colab T4）**：先 `pip uninstall -y faiss-cpu faiss`，再 `pip install -r requirements-gpu-cu12.txt`。訓練仍會寫出 **CPU 可讀** 的 `name_sales_faiss.index`。

**僅推論 / Web**：`pip install -r requirements-inference.txt`

## 訓練

**資料來源**（優先序）：`--data-csv 路徑` → 環境變數 `AMAZON_PRODUCTS_CSV` → 目前目錄 **`amazon_products.csv` 存在則直接讀** → 否則用 **KaggleHub** 下載。

```bash
cd amazon-name-sales-predictor
PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 250000
```

指定 CSV：

```bash
PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 250000 --data-csv ./amazon_products.csv
```

**整份 CSV 去掉非法列**（分塊重寫、預設備份 `.csv.bak`）：

```bash
PYTHONPATH=src python scripts/clean_amazon_products_csv.py --input amazon_products.csv
```

讀入訓練時也會再套同一套規則；`--sample-n 0` 為全量（很慢）。

## 預測

```bash
PYTHONPATH=src python -m amazon_name_sales_predictor.predict \
  --model-path models/name_sales_model.joblib \
  --name "Wireless Bluetooth Earbuds Noise Cancelling"
```

## Web

```bash
PYTHONPATH=src streamlit run src/amazon_name_sales_predictor/web.py
```

## Colab

見 **`COLAB.md`**。

## 注意

- 資料欄位會自動對應常見名稱（`title`、`boughtInLastMonth` 等）。
- 進階精度可再加入價格、評論數、星等作為特徵。
