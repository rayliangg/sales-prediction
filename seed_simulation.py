from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import kagglehub
import meilisearch
import pandas as pd
from kagglehub import KaggleDatasetAdapter


DATASET_REF = "thedevastator/product-prices-and-sizes-from-walmart-grocery"
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Walmart grocery data into Meilisearch.")
    parser.add_argument("--host", default="http://127.0.0.1:7700", help="Meilisearch host URL.")
    parser.add_argument("--api-key", default="masterKey", help="Meilisearch API key.")
    parser.add_argument("--index-name", default="walmart_products", help="Target Meilisearch index name.")
    parser.add_argument("--limit", type=int, default=20_000, help="Maximum documents to index.")
    parser.add_argument("--batch-size", type=int, default=1_000, help="Batch size for add_documents.")
    return parser.parse_args()


def find_primary_data_file(dataset_dir: Path) -> Path:
    candidates = sorted(dataset_dir.glob("**/*.csv")) + sorted(dataset_dir.glob("**/*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No csv/parquet data file found in {dataset_dir}")

    preferred = sorted(
        candidates,
        key=lambda p: ("product" not in p.name.lower(), "walmart" not in str(p).lower(), len(str(p))),
    )
    return preferred[0]


def load_walmart_dataframe(dataset_root: Path, dataset_file: Path) -> pd.DataFrame:
    file_path = str(dataset_file.relative_to(dataset_root))
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET_REF,
        file_path=file_path,
    )


def resolve_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def extract_number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = NUMBER_PATTERN.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def normalize_documents(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    columns = list(df.columns)
    name_col = resolve_column(columns, ["name", "title", "product_name", "product"])
    price_col = resolve_column(columns, ["price", "current_price", "sale_price", "final_price"])
    size_col = resolve_column(columns, ["size", "unit_size", "package_size"])
    brand_col = resolve_column(columns, ["brand", "brand_name"])
    category_col = resolve_column(columns, ["category", "primary_category", "department", "type"])
    rating_col = resolve_column(columns, ["avg_rating", "rating", "stars"])
    reviews_col = resolve_column(columns, ["total_reviews", "reviews", "review_count"])
    url_col = resolve_column(columns, ["product_url", "url"])
    image_col = resolve_column(columns, ["image", "image_url", "images"])
    currency_col = resolve_column(columns, ["currency"])

    if not name_col:
        raise ValueError("Cannot find a product name column.")

    working = df.copy()
    if limit > 0 and len(working) > limit:
        working = working.head(limit)

    docs: list[dict[str, Any]] = []
    for i, row in enumerate(working.itertuples(index=False), start=1):
        record = row._asdict()
        name = str(record.get(name_col, "")).strip()
        if not name:
            continue

        price_value = extract_number(record.get(price_col)) if price_col else None
        rating_value = extract_number(record.get(rating_col)) if rating_col else None
        reviews_value = extract_number(record.get(reviews_col)) if reviews_col else None

        doc = {
            "id": i,
            "name": name,
            "brand": str(record.get(brand_col, "")).strip() if brand_col else None,
            "category": str(record.get(category_col, "")).strip() if category_col else None,
            "size": str(record.get(size_col, "")).strip() if size_col else None,
            "price": price_value,
            "currency": str(record.get(currency_col, "USD")).strip() if currency_col else "USD",
            "avg_rating": rating_value,
            "review_count": int(reviews_value) if reviews_value is not None else None,
            "product_url": str(record.get(url_col, "")).strip() if url_col else None,
            "image_url": str(record.get(image_col, "")).strip() if image_col else None,
        }
        doc["search_text"] = " ".join(
            str(v) for v in [doc["name"], doc["brand"], doc["category"], doc["size"]] if v
        )
        docs.append(doc)
    return docs


def seed_meilisearch(
    host: str,
    api_key: str,
    index_name: str,
    docs: list[dict[str, Any]],
    batch_size: int,
) -> None:
    client = meilisearch.Client(host, api_key)
    index = client.index(index_name)

    index.update_filterable_attributes(["category", "brand", "currency"])
    index.update_sortable_attributes(["price", "avg_rating", "review_count"])
    index.update_searchable_attributes(["name", "brand", "category", "size", "search_text"])

    for offset in range(0, len(docs), batch_size):
        batch = docs[offset : offset + batch_size]
        index.add_documents(batch, primary_key="id")


def main() -> None:
    args = parse_args()

    dataset_dir = Path(kagglehub.dataset_download(DATASET_REF))
    data_file = find_primary_data_file(dataset_dir)
    df = load_walmart_dataframe(dataset_dir, data_file)
    docs = normalize_documents(df, limit=args.limit)
    seed_meilisearch(args.host, args.api_key, args.index_name, docs, args.batch_size)

    print(f"Loaded data file: {data_file}")
    print(f"Indexed documents: {len(docs)}")
    print(f"Meilisearch index: {args.index_name}")


if __name__ == "__main__":
    main()
