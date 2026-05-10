from __future__ import annotations

import argparse
from pathlib import Path

import hnswlib
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import download_dataset, load_dataset_frame, prepare_training_frame


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000)),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )


def train(output_dir: Path, sample_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

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
    x_all_dense = svd.fit_transform(x_all).astype(np.float32)

    hnsw = hnswlib.Index(space="cosine", dim=svd_dim)
    hnsw.init_index(max_elements=x_all_dense.shape[0], ef_construction=200, M=32)
    labels = np.arange(x_all_dense.shape[0])
    hnsw.add_items(x_all_dense, labels)
    hnsw.set_ef(100)

    index_file = output_dir / "name_sales_hnsw.bin"
    hnsw.save_index(str(index_file))

    artifact = {
        "pipeline": pipe,
        "svd": svd,
        "hnsw_index_file": index_file.name,
        "hnsw_space": "cosine",
        "hnsw_dim": svd_dim,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_n = args.sample_n if args.sample_n > 0 else None
    train(Path(args.output_dir), sample_n=sample_n)

