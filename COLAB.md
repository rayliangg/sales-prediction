# Colab Training Guide

這份指南讓你在 Google Colab 訓練模型，並把模型檔存到 Google Drive。

## 1) 開啟 Colab

建立一個新的 Python Notebook，執行以下區塊。

## 2) 啟用 GPU（建議）

在 Colab：**執行階段 → 變更執行階段類型 → 硬體加速器選 T4 GPU**。  
這樣會安裝 `faiss-gpu`，訓練時若偵測到 GPU 會用 GPU 建 FAISS 索引（失敗則自動改 CPU）。

## 3) 安裝套件與抓專案

```python
!git clone https://github.com/<your-account>/<your-repo>.git
%cd amazon-name-sales-predictor
!pip install -r requirements.txt
```

> 如果你還沒把專案推到 GitHub，也可以先把整個 `amazon-name-sales-predictor` 資料夾上傳到 Colab 左側 Files。

## 4) 掛載 Google Drive（用來保存模型）

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 5) 在 Colab 訓練

```python
!PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 250000
```

全量訓練（較慢）：

```python
!PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models --sample-n 0
```

## 6) 複製模型到 Google Drive

```python
!mkdir -p /content/drive/MyDrive/amazon-name-sales-models
!cp models/name_sales_model.joblib /content/drive/MyDrive/amazon-name-sales-models/
!cp models/name_sales_faiss.index /content/drive/MyDrive/amazon-name-sales-models/
```

## 7) 回到本機使用

把兩個檔案下載或同步回你的本機專案 `models/`：
- `name_sales_model.joblib`
- `name_sales_faiss.index`

索引檔是 **CPU 版 FAISS 索引**（訓練時即使用 GPU 建索引，也已轉成 CPU 再存檔），本機推論不需 GPU。

本機若只做 Web / 預測，可只裝較輕的依賴：

```bash
pip install -r requirements-inference.txt
```

然後啟動 Web：

```bash
PYTHONPATH=src streamlit run src/amazon_name_sales_predictor/web.py
```

