from __future__ import annotations

from pathlib import Path
from typing import Iterable

import kagglehub
import pandas as pd


DATASET_REF = "asaniczka/amazon-products-dataset-2023-1-4m-products"

NAME_CANDIDATES = ["title", "product_name", "name"]
TARGET_CANDIDATES = ["boughtInLastMonth", "monthly_sales", "units_sold", "sales"]
CATEGORY_CANDIDATES = [
    "category_name",
    "main_category",
    "category",
    "categoryName",
    "category_id",
]


def download_dataset() -> Path:
    """Download dataset via kagglehub and return local path."""
    path = kagglehub.dataset_download(DATASET_REF)
    return Path(path)


def _find_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def load_dataset_frame(dataset_dir: Path, sample_n: int | None = 250_000) -> pd.DataFrame:
    """Load the best csv/parquet candidate for model training."""
    files = sorted(dataset_dir.glob("**/*.parquet")) + sorted(dataset_dir.glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv/parquet file found in {dataset_dir}")

    file_path = _pick_best_data_file(files)
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)

    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    return df


def _pick_best_data_file(files: list[Path]) -> Path:
    """Choose file likely to contain product names and sales target."""
    best_score = float("-inf")
    best_file = files[0]

    for file_path in files:
        try:
            if file_path.suffix == ".parquet":
                probe = pd.read_parquet(file_path).head(5)
            else:
                probe = pd.read_csv(file_path, nrows=5, low_memory=False)
            cols = list(probe.columns)
        except Exception:
            continue

        name_col = _find_column(cols, NAME_CANDIDATES)
        target_col = _find_column(cols, TARGET_CANDIDATES)
        category_col = _find_column(cols, CATEGORY_CANDIDATES)

        # Prioritize files that have required columns; tie-break by column richness.
        score = 0
        score += 100 if name_col else 0
        score += 100 if target_col else 0
        score += 30 if category_col else 0
        score += len(cols) * 0.1

        if score > best_score:
            best_score = score
            best_file = file_path

    return best_file


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    """Map source dataframe into [name, category, sales] training schema."""
    name_col = _find_column(df.columns, NAME_CANDIDATES)
    target_col = _find_column(df.columns, TARGET_CANDIDATES)
    category_col = _find_column(df.columns, CATEGORY_CANDIDATES)

    if not name_col or not target_col:
        raise ValueError(
            "Could not locate required columns. "
            f"Need name from {NAME_CANDIDATES}, target from {TARGET_CANDIDATES}."
        )

    if not category_col:
        category_col = "__category__"
        df[category_col] = "UNKNOWN"

    out = pd.DataFrame(
        {
            "name": df[name_col].astype(str),
            "category": df[category_col].astype(str),
            "sales": _to_numeric(df[target_col]),
        }
    )

    out = out.dropna(subset=["name", "sales"])
    out = out[out["name"].str.len() > 3]
    out = out[out["sales"] >= 0]
    return out, name_col, category_col, target_col


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")

