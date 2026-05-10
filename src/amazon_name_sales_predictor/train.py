from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import (
    download_dataset,
    load_dataset_csv,
    load_dataset_frame,
    prepare_training_frame,
    resolve_local_csv_path,
)


def _faiss_gpu_count() -> int:
    try:
        return int(faiss.get_num_gpus())
    except Exception:
        return 0


def _build_faiss_hnsw_index(x_all_dense: np.ndarray, svd_dim: int, m: int = 32) -> tuple[faiss.Index, bool]:
    """Build HNSW index. Optional GPU `add`; always returns a CPU index for `write_index` / CPU inference."""
    index_cpu = faiss.IndexHNSWFlat(svd_dim, m, faiss.METRIC_INNER_PRODUCT)
    index_cpu.hnsw.efConstruction = 200
    index_cpu.hnsw.efSearch = 100

    if _faiss_gpu_count() > 0:
        try:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)
            gpu_index.add(x_all_dense)
            out = faiss.index_gpu_to_cpu(gpu_index)
            return out, True
        except Exception as exc:
            print(f"[faiss] GPU index build failed ({exc}); using CPU.")

    index_cpu.add(x_all_dense)
    return index_cpu, False


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000)),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def train(output_dir: Path, sample_n: int, data_csv: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_local_csv_path(data_csv)
    if csv_path is not None:
        print(f"Loading data from CSV: {csv_path}")
        raw_df = load_dataset_csv(csv_path, sample_n=sample_n)
    else:
        if data_csv is not None:
            raise FileNotFoundError(f"--data-csv not found: {data_csv}")
        print("Downloading dataset (KaggleHub)...")
        dataset_dir = download_dataset()
        raw_df = load_dataset_frame(dataset_dir, sample_n=sample_n)

    df, src_name, src_category, src_target = prepare_training_frame(raw_df)

    x_train, x_test, y_train, y_test = train_test_split(
        df["name"], np.log1p(df["sales"]), test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(x_train, y_train)

    pred_test = np.expm1(pipe.predict(x_test))
    true_test = np.expm1(y_test)
    mae = mean_absolute_error(true_test, pred_test)

    tfidf = pipe.named_steps["tfidf"]
    x_all = tfidf.transform(df["name"])
    svd_dim = min(256, max(2, x_all.shape[1] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    x_all_dense = np.ascontiguousarray(svd.fit_transform(x_all), dtype=np.float32)
    faiss.normalize_L2(x_all_dense)

    # HNSW graph on unit vectors with inner product = cosine similarity search.
    index, used_gpu = _build_faiss_hnsw_index(x_all_dense, svd_dim, m=32)

    index_file = output_dir / "name_sales_faiss.index"
    faiss.write_index(index, str(index_file))

    artifact = {
        "pipeline": pipe,
        "svd": svd,
        "faiss_index_file": index_file.name,
        "faiss_dim": svd_dim,
        "faiss_metric": "inner_product_on_l2_normalized",
        "faiss_trained_on_gpu": used_gpu,
        "train_names": df["name"].tolist(),
        "train_sales": df["sales"].tolist(),
        "train_categories": df["category"].tolist(),
        "metrics": {"test_mae": float(mae), "train_size": int(len(df))},
        "source_columns": {
            "name_column": src_name,
            "category_column": src_category,
            "target_column": src_target,
        },
    }
    out_file = output_dir / "name_sales_model.joblib"
    joblib.dump(artifact, out_file)

    print(f"Model saved to: {out_file}")
    print(f"FAISS index saved to: {index_file} (GPU build: {used_gpu})")
    print(f"Test MAE: {mae:.2f}")
    print(f"Training rows: {len(df)}")
    print(f"Source columns -> name: {src_name}, category: {src_category}, sales: {src_target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train product-name -> sales model from Amazon dataset."
    )
    parser.add_argument("--output-dir", default="models", help="Directory for model artifact.")
    parser.add_argument(
        "--sample-n",
        type=int,
        default=250000,
        help="Rows to sample for faster training; use 0 for full dataset.",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="Local products CSV. If omitted, uses AMAZON_PRODUCTS_CSV env or ./amazon_products.csv when present; else KaggleHub.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_n = args.sample_n if args.sample_n > 0 else None
    train(Path(args.output_dir), sample_n=sample_n, data_csv=args.data_csv)
