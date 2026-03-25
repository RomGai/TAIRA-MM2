from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def _parse_meta_line(line: str) -> dict:
    text = line.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def prepare_meta(raw_meta_path: Path, metadata_csv_path: Path, output_path: Path) -> dict:
    metadata_df = pd.read_csv(metadata_csv_path, dtype={"id": str})
    metadata_df["id"] = metadata_df["id"].astype(str).str.strip()
    metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce")

    valid_ids = set(metadata_df["id"].tolist())
    price_map = {
        row["id"]: None if pd.isna(row["price"]) else float(row["price"])
        for _, row in metadata_df.iterrows()
    }

    kept = []
    total = 0
    dropped = 0
    assigned_price = 0

    with raw_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = _parse_meta_line(line)
            asin = str(rec.get("asin", "")).strip()
            if not asin or asin not in valid_ids:
                dropped += 1
                continue
            if rec.get("price") in (None, "", "NaN"):
                p = price_map.get(asin)
                if p is not None:
                    rec["price"] = p
                    assigned_price += 1
            kept.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "raw_total": total,
        "kept": len(kept),
        "dropped": dropped,
        "assigned_price": assigned_price,
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter beauty raw meta by metadata.csv ids and backfill price.")
    parser.add_argument("--raw-meta", required=True, help="Path to raw meta_Beauty.json (jsonl / python-dict lines)")
    parser.add_argument("--metadata-csv", default="data/amazon_beauty/metadata.csv")
    parser.add_argument("--output", default="data/amazon_beauty/meta_Beauty.filtered.jsonl")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summary = prepare_meta(Path(args.raw_meta), Path(args.metadata_csv), Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
