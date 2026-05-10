"""Keep only columns used by training (name, category, sales). Chunked IO for large files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEEP = ["title", "category_id", "boughtInLastMonth"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("amazon_products.csv"))
    parser.add_argument("--output", type=Path, default=None, help="Default: <input>.slim.tmp then replace input")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Missing: {inp}")

    out = args.output
    if out is None:
        out = inp.with_suffix(".slim.tmp")
    else:
        out = out.resolve()

    first = True
    total = 0
    for chunk in pd.read_csv(inp, usecols=KEEP, chunksize=args.chunksize, low_memory=False):
        chunk.to_csv(out, mode="w" if first else "a", index=False, header=first)
        total += len(chunk)
        first = False
        print(f"rows: {total}", flush=True)

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
