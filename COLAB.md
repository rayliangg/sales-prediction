# Colab Training

`REPO_DIR` 須與 GitHub **repo 名稱**一致（例如 `sales-prediction`）。`--sample-n` 可自行改（`0` = 全量，很慢）。

### 資料從哪裡讀（與本機相同）

優先序：`--data-csv` → 環境變數 `AMAZON_PRODUCTS_CSV` → **`{REPO_DIR}/amazon_products.csv` 若存在** → 否則 **KaggleHub** 下載。

在 Colab 可把 **`amazon_products.csv`** 上傳到 **`/content/sales-prediction/`**（與 `src/` 同層），再跑下方訓練，即會自動讀 CSV、**不會**再下載資料集。

上傳後若要**整份 CSV 去掉非法列**（分塊重寫），在該目錄執行：

`PYTHONPATH=src python scripts/clean_amazon_products_csv.py --input amazon_products.csv`

---

## CPU 版（`faiss-cpu`，執行階段選 **CPU** 或 **T4** 皆可；FAISS 在 CPU 建索引）

```python
import os
import subprocess
import sys

REPO_URL = "https://github.com/rayliangg/sales-prediction.git"
REPO_DIR = "sales-prediction"
SAMPLE_N = 250_000  # 全量改 0

root, target = "/content", f"/content/{REPO_DIR}"

def sh(*a, **k):
    subprocess.run(a, **k, check=True)

if os.path.isdir(f"{target}/.git"):
    sh("git", "-C", target, "pull")
elif os.path.isdir(target):
    raise RuntimeError(f"請先刪除或改名：{target}")
else:
    sh("git", "clone", REPO_URL, cwd=root)

os.chdir(target)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "-r", "requirements.txt"],
    check=True,
)

env = {**os.environ, "PYTHONPATH": "src"}
train_cmd = [
    sys.executable,
    "-m",
    "amazon_name_sales_predictor.train",
    "--output-dir",
    "models",
    "--sample-n",
    str(SAMPLE_N),
]
# 若已上傳 amazon_products.csv 到本專案根目錄，可改為：
# train_cmd += ["--data-csv", "amazon_products.csv"]
subprocess.run(train_cmd, cwd=target, env=env, check=True)
print("完成：models/name_sales_model.joblib 與 models/name_sales_faiss.index")
```

---

## GPU 版（`faiss-gpu-cu12`，執行階段請選 **T4 GPU**；FAISS 建索引用 GPU）

```python
import os
import subprocess
import sys

REPO_URL = "https://github.com/rayliangg/sales-prediction.git"
REPO_DIR = "sales-prediction"
SAMPLE_N = 250_000  # 全量改 0

root, target = "/content", f"/content/{REPO_DIR}"

def sh(*a, **k):
    subprocess.run(a, **k, check=True)

if os.path.isdir(f"{target}/.git"):
    sh("git", "-C", target, "pull")
elif os.path.isdir(target):
    raise RuntimeError(f"請先刪除或改名：{target}")
else:
    sh("git", "clone", REPO_URL, cwd=root)

os.chdir(target)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "-r", "requirements.txt"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu", "faiss"],
    check=False,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "-r", "requirements-gpu-cu12.txt"],
    check=True,
)

import faiss

assert faiss.get_num_gpus() >= 1, "請確認：執行階段 → 變更執行階段類型 → T4 GPU"

env = {**os.environ, "PYTHONPATH": "src"}
train_cmd = [
    sys.executable,
    "-m",
    "amazon_name_sales_predictor.train",
    "--output-dir",
    "models",
    "--sample-n",
    str(SAMPLE_N),
]
# 同上，有本機 CSV 時可：train_cmd += ["--data-csv", "amazon_products.csv"]
subprocess.run(train_cmd, cwd=target, env=env, check=True)
print("完成：models/name_sales_model.joblib 與 models/name_sales_faiss.index")
```

---

## Google Drive（可選）

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
!mkdir -p /content/drive/MyDrive/amazon-name-sales-models
!cp /content/sales-prediction/models/name_sales_model.joblib /content/sales-prediction/models/name_sales_faiss.index /content/drive/MyDrive/amazon-name-sales-models/
```

（若 `REPO_DIR` 不是 `sales-prediction`，請改路徑。）

## 本機

下載兩個檔到本機 `models/`，依 **README** 用 `requirements-inference.txt` 跑預測或 Streamlit。
