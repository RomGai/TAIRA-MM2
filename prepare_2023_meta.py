from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_meta_line(line: str) -> dict[str, Any]:
    text = line.strip()
    if not text:
        return {}
    return json.loads(text)


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict, tuple, set)) and not value:
            continue
        return value
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_normalize_text(x) for x in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            text = _normalize_text(v)
            if text:
                parts.append(f"{k}: {text}")
        return "; ".join(parts).strip()
    return str(value).strip()


def _normalize_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned:
            return None
        value = cleaned
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _normalize_categories(raw_categories: Any, fallback_category: Any = None) -> list[list[str]]:
    source = raw_categories
    if source is None or source == "":
        source = fallback_category

    if isinstance(source, list):
        if not source:
            return []
        if all(isinstance(x, str) for x in source):
            segs = [x.strip() for x in source if str(x).strip()]
            return [segs] if segs else []

        out: list[list[str]] = []
        for path in source:
            if isinstance(path, list):
                segs = [str(x).strip() for x in path if str(x).strip()]
                if segs:
                    out.append(segs)
            elif isinstance(path, str) and path.strip():
                segs = [seg.strip() for seg in path.split("|") if seg.strip()]
                if segs:
                    out.append(segs)
        return out

    if isinstance(source, str) and source.strip():
        normalized = source.replace(">", "|")
        segs = [seg.strip() for seg in normalized.split("|") if seg.strip()]
        return [segs] if segs else []

    return []


def _extract_image_url(record: dict[str, Any]) -> str:
    images = record.get("images")
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict):
                url = _first_non_empty(image.get("hi_res"), image.get("large"), image.get("thumb"), image.get("url"))
                if isinstance(url, str) and url.strip():
                    return url.strip()
            elif isinstance(image, str) and image.strip():
                return image.strip()

    direct = _first_non_empty(record.get("imUrl"), record.get("image"), record.get("image_url"))
    return direct.strip() if isinstance(direct, str) else ""


def _extract_related(record: dict[str, Any]) -> dict[str, Any]:
    related = record.get("related")
    if isinstance(related, dict):
        return {k: v for k, v in related.items() if v not in (None, [], {}, "")}

    out: dict[str, Any] = {}
    for key in ("also_bought", "also_viewed", "bought_together", "buy_after_viewing"):
        value = record.get(key)
        if value not in (None, [], {}, ""):
            out[key] = value
    return out


def _normalize_sales_rank(record: dict[str, Any], metadata_row: dict[str, Any] | None) -> dict[str, Any]:
    sales_rank = record.get("salesRank")
    if isinstance(sales_rank, dict) and sales_rank:
        return sales_rank

    if metadata_row:
        ranking = metadata_row.get("ranking")
        category = _normalize_text(_first_non_empty(record.get("main_category"), metadata_row.get("category")))
        if pd.notna(ranking):
            try:
                ranking_value = int(float(ranking))
            except (TypeError, ValueError):
                ranking_value = None
            if ranking_value is not None and category:
                return {category: ranking_value}
    return {}


def _load_metadata(metadata_csv_path: Path) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, float | None]]:
    metadata_df = pd.read_csv(metadata_csv_path, dtype={"id": str})
    metadata_df["id"] = metadata_df["id"].astype(str).str.strip()
    if "price" in metadata_df.columns:
        metadata_df["price"] = pd.to_numeric(metadata_df["price"], errors="coerce")

    valid_ids = set(metadata_df["id"].tolist())
    metadata_rows = {
        row["id"]: row.to_dict()
        for _, row in metadata_df.iterrows()
    }
    price_map = {
        row["id"]: None if "price" not in row or pd.isna(row["price"]) else float(row["price"])
        for _, row in metadata_df.iterrows()
    }
    return valid_ids, metadata_rows, price_map


def _canonicalize_record(record: dict[str, Any], item_id: str, metadata_row: dict[str, Any] | None, fallback_price: float | None) -> tuple[dict[str, Any], bool]:
    normalized = dict(record)

    normalized["asin"] = item_id
    if "parent_asin" not in normalized:
        normalized["parent_asin"] = item_id

    title = _normalize_text(_first_non_empty(record.get("title"), metadata_row.get("title") if metadata_row else None))
    if title:
        normalized["title"] = title

    description = _normalize_text(_first_non_empty(record.get("description"), metadata_row.get("description") if metadata_row else None, record.get("features")))
    normalized["description"] = description

    categories = _normalize_categories(record.get("categories"), metadata_row.get("category") if metadata_row else None)
    normalized["categories"] = categories

    image_url = _extract_image_url(record)
    if image_url:
        normalized["imUrl"] = image_url

    related = _extract_related(record)
    if related:
        normalized["related"] = related

    price = _normalize_price(record.get("price"))
    assigned_price = False
    if price is None and fallback_price is not None:
        price = fallback_price
        assigned_price = True
    if price is not None:
        normalized["price"] = price

    sales_rank = _normalize_sales_rank(record, metadata_row)
    if sales_rank:
        normalized["salesRank"] = sales_rank

    return normalized, assigned_price


def prepare_meta(raw_meta_path: Path, metadata_csv_path: Path, output_path: Path) -> dict[str, Any]:
    valid_ids, metadata_rows, price_map = _load_metadata(metadata_csv_path)

    kept: list[dict[str, Any]] = []
    total = 0
    dropped = 0
    assigned_price = 0

    with raw_meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = _parse_meta_line(line)
            item_id = _normalize_text(
                _first_non_empty(
                    record.get("asin"),
                    record.get("parent_asin"),
                    record.get("id"),
                    record.get("item_id"),
                )
            )
            if not item_id or item_id not in valid_ids:
                dropped += 1
                continue

            normalized, price_was_assigned = _canonicalize_record(
                record=record,
                item_id=item_id,
                metadata_row=metadata_rows.get(item_id),
                fallback_price=price_map.get(item_id),
            )
            if price_was_assigned:
                assigned_price += 1
            kept.append(normalized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "raw_total": total,
        "kept": len(kept),
        "dropped": dropped,
        "assigned_price": assigned_price,
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize 2023 Amazon meta jsonl into the legacy filtered-meta format used by downstream pipelines."
    )
    parser.add_argument("--raw-meta", required=True, help="Path to raw 2023 meta jsonl")
    parser.add_argument("--metadata-csv", default="data/amazon_music/metadata.csv")
    parser.add_argument("--output", default="data/amazon_music/meta_2023.filtered.jsonl")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summary = prepare_meta(Path(args.raw_meta), Path(args.metadata_csv), Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
