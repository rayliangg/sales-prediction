# Colab Training Guide

這份指南讓你在 Google Colab 訓練模型，並把模型檔存到 Google Drive。

## 1) 開啟 Colab

建立一個新的 Python Notebook，執行以下區塊。

## 2) 硬體：CPU 或 GPU FAISS

- **執行階段 → 變更執行階段類型 → T4 GPU**：若要 **GPU 版 FAISS 建索引**，請務必選 GPU；僅裝 `faiss-cpu` 時不會用到 GPU。
- **sklearn**（TF-IDF、Ridge、SVD）在 Colab 仍主要跑在 **CPU**；GPU 主要加速 **FAISS 建索引**（有裝 `faiss-gpu-cu12` 且偵測到 GPU 時）。

Colab 為 **Python 3.12** 時，請用 **`faiss-gpu-cu12`**（見下一節），不要用 PyPI 的 `faiss-gpu`（常無 wheel）。

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

### 3b) 要用 GPU 版 FAISS 訓練（Colab T4 / CUDA 12）

先完成上一格（已在 `sales-prediction` 目錄、已裝 `requirements.txt`）。再執行：**解除 `faiss-cpu`**，改裝 **`faiss-gpu-cu12`**（與 Python 3.12 相容的 CUDA 12 wheel）。

```python
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss"], check=False)

p = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "-r", "requirements-gpu-cu12.txt"],
    cwd=os.getcwd(),
    capture_output=True,
    text=True,
)
print(p.stdout or "")
print(p.stderr or "")
if p.returncode != 0:
    raise RuntimeError("faiss-gpu-cu12 安裝失敗，請把上方錯誤貼出排查。")

import faiss
print("faiss.get_num_gpus() =", faiss.get_num_gpus())
```

預期在已選 **T4** 的情況下，`faiss.get_num_gpus()` 至少為 **1**。接著跑訓練（第 5 節）時，`train.py` 會在 GPU 上建 FAISS 索引再存成 CPU 索引檔。

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

