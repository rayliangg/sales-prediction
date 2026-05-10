# Colab Training Guide

這份指南讓你在 Google Colab 訓練模型，並把模型檔存到 Google Drive。

## 1) 開啟 Colab

建立一個新的 Python Notebook，執行以下區塊。

## 2) 安裝套件與抓專案

```python
!git clone https://github.com/<your-account>/<your-repo>.git
%cd amazon-name-sales-predictor
!pip install -r requirements.txt
```

> 如果你還沒把專案推到 GitHub，也可以先把整個 `amazon-name-sales-predictor` 資料夾上傳到 Colab 左側 Files。

## 3) 掛載 Google Drive（用來保存模型）

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4) 在 Colab 訓練

```python
!PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 250000
```

全量訓練（較慢）：

```python
!PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 0
```

## 5) 複製模型到 Google Drive

```python
!mkdir -p /content/drive/MyDrive/amazon-name-sales-models
!cp models/name_sales_model.joblib /content/drive/MyDrive/amazon-name-sales-models/
!cp models/name_sales_hnsw.bin /content/drive/MyDrive/amazon-name-sales-models/
```

## 6) 回到本機使用

把兩個檔案下載或同步回你的本機專案 `models/`：
- `name_sales_model.joblib`
- `name_sales_hnsw.bin`

然後啟動 Web：

```bash
PYTHONPATH=src streamlit run src/amazon_name_sales_predictor/web.py
```

