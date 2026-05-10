#!/usr/bin/env python3
"""Rewrite CSV in chunks, dropping invalid rows (same rules as data.filter_invalid_product_rows)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd
from amazon_name_sales_predictor.data import filter_invalid_product_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("amazon_products.csv"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: rewrite input via .clean.tmp + backup .bak",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing: {inp}")

    out = args.output or inp.with_suffix(".clean.tmp")
    out = out.resolve()

    first = True
    total_in = 0
    total_out = 0
    for chunk in pd.read_csv(inp, chunksize=args.chunksize, low_memory=False):
        total_in += len(chunk)
        clean = filter_invalid_product_rows(chunk)
        total_out += len(clean)
        clean.to_csv(out, mode="w" if first else "a", index=False, header=first)
        first = False
        print(f"read={total_in} written={total_out}", flush=True)

    dropped = total_in - total_out
    print(f"Done. dropped_rows={dropped} kept_rows={total_out}")

    if args.output is None:
        bak = inp.with_suffix(".csv.bak")
        if bak.exists():
            bak.unlink()
        inp.rename(bak)
        out.rename(inp)
        print(f"Replaced {inp.name}; backup: {bak.name}")
    else:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
