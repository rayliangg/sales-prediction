# Colab Training Guide

這份指南讓你在 Google Colab 訓練模型，並把模型檔存到 Google Drive。

## 1) 開啟 Colab

建立一個新的 Python Notebook，執行以下區塊。

## 2) 啟用 GPU（建議）

在 Colab：**執行階段 → 變更執行階段類型 → 硬體加速器選 T4 GPU**（可選；sklearn 與預設 **faiss-cpu** 仍主要吃 CPU）。  
**注意**：Colab 目前多為 **Python 3.12**，PyPI 上的 **`faiss-gpu` 通常沒有 wheel**（會出現 `No matching distribution found`）。本專案 **`requirements.txt` 已預設 `faiss-cpu`**，可直接安裝成功。

## 3) 安裝套件與抓專案

`git clone` 後的資料夾名稱 = **GitHub 上的 repo 名稱**（例如 `sales-prediction`），**不是**本機資料夾名 `amazon-name-sales-predictor`。  
請把下面 `REPO_URL` 改成你的倉庫網址；`REPO_DIR` 請與網址最後一段 repo 名一致。

```python
import os
import subprocess

REPO_URL = "https://github.com/rayliangg/sales-prediction.git"  # 改成你的
REPO_DIR = "sales-prediction"  # 必須與 GitHub repo 名相同（clone 出來的資料夾名）

root = "/content"
target = os.path.join(root, REPO_DIR)

def run(cmd, cwd=None):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd)
    subprocess.run(cmd, cwd=cwd, check=True)


def pip_install(req_file: str) -> int:
    import sys

    print("+", sys.executable, "-m pip install -r", req_file)
    p = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "-r", req_file],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
    )
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr)
    return p.returncode


if os.path.isdir(os.path.join(target, ".git")):
    run(["git", "-C", target, "pull"])
elif os.path.isdir(target):
    raise RuntimeError(
        f"{target} 已存在但不是 git 倉庫。請在 Colab 執行：!rm -rf {target} 後再重跑，或改 REPO_DIR。"
    )
else:
    run(["git", "clone", REPO_URL], cwd=root)

os.chdir(target)
print("目前目錄:", os.getcwd())

if pip_install("requirements.txt") != 0:
    print("\n=== requirements.txt 失敗，改試 requirements-inference.txt ===\n")
    if not os.path.isfile("requirements-inference.txt") or pip_install("requirements-inference.txt") != 0:
        raise RuntimeError("pip install 仍失敗，請把上方 pip 錯誤訊息貼出來排查。")
```

若你確定要**刪掉舊目錄重新 clone**（會刪除該資料夾內所有檔案）：

```python
!rm -rf /content/sales-prediction
```

再把上面「抓專案」區塊重跑一次（`REPO_DIR` 請與你要刪的資料夾名一致）。

> 如果你還沒把專案推到 GitHub，也可以先把整個專案資料夾上傳到 Colab 左側 Files，再 `%cd` 到該資料夾後執行 `pip install -r requirements.txt`。

## 4) 掛載 Google Drive（用來保存模型）

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 5) 在 Colab 訓練

請確認已在 repo 根目錄（內有 `requirements.txt`、`src/`）。若上一格已 `os.chdir`，通常可直接跑；否則先：

```python
%cd /content/sales-prediction
```

（路徑請改成你的 `REPO_DIR`。）

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

