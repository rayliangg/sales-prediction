from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import joblib
import numpy as np


def predict_one(model_path: Path, product_name: str, top_k: int = 20) -> dict:
    artifact = joblib.load(model_path)
    if "faiss_index_file" not in artifact:
        raise ValueError(
            "This model was trained with an older index format. "
            "Re-train with: PYTHONPATH=src python -m amazon_name_sales_predictor.train --output-dir models"
        )

    pipeline = artifact["pipeline"]
    svd = artifact["svd"]
    train_names = artifact["train_names"]
    train_sales = np.array(artifact["train_sales"], dtype=float)
    train_categories = np.array(artifact["train_categories"], dtype=object)
    faiss_dim = int(artifact["faiss_dim"])
    faiss_index_file = artifact["faiss_index_file"]

    pred_sales = float(np.expm1(pipeline.predict([product_name]))[0])
    tfidf = pipeline.named_steps["tfidf"]
    vec_sparse = tfidf.transform([product_name])
    vec_dense = np.ascontiguousarray(svd.transform(vec_sparse), dtype=np.float32)
    faiss.normalize_L2(vec_dense)

    index_path = model_path.parent / faiss_index_file
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    top_k = max(1, min(int(top_k), len(train_names)))
    index.hnsw.efSearch = max(100, top_k)

    similarities, indices = index.search(vec_dense, top_k)
    idx = indices[0]
    sim_row = similarities[0]

    neighbor_sales = train_sales[idx]
    neighbor_categories = train_categories[idx]
    dominant_category = _mode(neighbor_categories.tolist())

    # Normalization: compare with similar products only.
    mean_sales = float(np.mean(neighbor_sales))
    std_sales = float(np.std(neighbor_sales)) if float(np.std(neighbor_sales)) > 1e-9 else 1.0
    z_score = (pred_sales - mean_sales) / std_sales
    percentile = float((neighbor_sales < pred_sales).mean() * 100.0)

    neighbors = []
    for rank, i in enumerate(idx, start=1):
        sim = float(sim_row[rank - 1])
        neighbors.append(
            {
                "rank": rank,
                "name": train_names[int(i)],
                "sales": float(train_sales[int(i)]),
                "category": str(train_categories[int(i)]),
                "distance": -sim,
            }
        )

    return {
        "input_name": product_name,
        "predicted_sales": pred_sales,
        "similar_products_avg_sales": mean_sales,
        "similar_products_std_sales": std_sales,
        "normalized_z_score": z_score,
        "normalized_percentile_in_similars": percentile,
        "inferred_category": dominant_category,
        "top_similar_examples": neighbors,
    }


def _mode(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sales from product name and normalize among similar products."
    )
    parser.add_argument("--model-path", default="models/name_sales_model.joblib")
    parser.add_argument("--name", required=True, help="Input product name")
    parser.add_argument("--top-k", type=int, default=20, help="Number of similar products")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_one(Path(args.model_path), args.name, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
