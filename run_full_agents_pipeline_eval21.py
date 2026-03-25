"""Run Amazon full agents pipeline with per-user eval item catalogs.

Key behavior:
- For each evaluation user, build a (1 positive + N negatives) catalog BEFORE
  Agent 1, and use it as Agent 1's full item input.
- Agent 2 keeps normal full-history processing.
- After each user is processed, print cumulative metrics using the same grouped
  ranking metrics style as `test/microlens/test_with_llava.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Set, Tuple

from run_full_agents_pipeline import build_argparser as build_full_argparser
from run_full_agents_pipeline import run_pipeline


@dataclass
class EvalUnit:
    user_id: str
    pos_items: List[str]
    neg_items: List[str]


def _read_user_items_negs(path: str | Path) -> List[EvalUnit]:
    rows: List[EvalUnit] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        has_header = first_line.lower().startswith("user_id\t")
        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                user_id = str(row.get("user_id", "")).strip()
                pos = [x.strip() for x in str(row.get("pos", "")).split(",") if x.strip()]
                neg = [x.strip() for x in str(row.get("neg", "")).split(",") if x.strip()]
                if user_id and pos:
                    rows.append(EvalUnit(user_id=user_id, pos_items=pos, neg_items=neg))
        else:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                user_id = str(row[0]).strip()
                pos = [x.strip() for x in str(row[1]).split(",") if x.strip()]
                neg = [x.strip() for x in str(row[2]).split(",") if x.strip()]
                if user_id and pos:
                    rows.append(EvalUnit(user_id=user_id, pos_items=pos, neg_items=neg))
    return rows


def _write_user_items_negs_for_single_user(
    rows: Sequence[EvalUnit],
    user_id: str,
    out_path: str | Path,
) -> int:
    """Write all pos/neg rows for one user to a temp Agent2 input TSV."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    selected = [r for r in rows if r.user_id == str(user_id)]
    if not selected:
        raise ValueError(f"No rows found for user_id={user_id} in Agent2 user-items-negs input")

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "pos", "neg"], delimiter="\t")
        writer.writeheader()
        for r in selected:
            writer.writerow(
                {
                    "user_id": r.user_id,
                    "pos": ",".join(r.pos_items),
                    "neg": ",".join(r.neg_items),
                }
            )
    return len(selected)


def _read_item_desc_rows(path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            rows.append(
                {
                    "item_id": item_id,
                    "image": str(row.get("image", "")),
                    "summary": str(row.get("summary", "")),
                }
            )
    return rows


def _user_seen_items(user_pairs_tsv: str | Path, user_id: str) -> Set[str]:
    seen: Set[str] = set()
    with Path(user_pairs_tsv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if str(row.get("user_id", "")).strip() != str(user_id):
                continue
            item_id = str(row.get("item_id", "")).strip()
            if item_id:
                seen.add(item_id)
    return seen


def _latest_positive_by_timestamp(
    user_pairs_tsv: str | Path,
    user_id: str,
    positive_item_ids: Sequence[str],
) -> str:
    """Pick the latest (max timestamp) positive item for the user.

    Falls back to the first positive item when no valid timestamp can be found.
    """
    if not positive_item_ids:
        raise ValueError(f"No positive items for user={user_id}")

    target_set = {str(i).strip() for i in positive_item_ids if str(i).strip()}
    if not target_set:
        return str(positive_item_ids[0]).strip()

    latest_item = None
    latest_ts = None
    with Path(user_pairs_tsv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if str(row.get("user_id", "")).strip() != str(user_id):
                continue
            item_id = str(row.get("item_id", "")).strip()
            if item_id not in target_set:
                continue
            ts_raw = str(row.get("timestamp", "")).strip()
            try:
                ts_val = int(ts_raw)
            except (TypeError, ValueError):
                continue
            if latest_ts is None or ts_val > latest_ts:
                latest_ts = ts_val
                latest_item = item_id

    return latest_item or str(positive_item_ids[0]).strip()


def _build_eval21_catalog(
    all_item_ids: Sequence[str],
    unit: EvalUnit,
    chosen_positive: str,
    user_seen_item_ids: Set[str],
    seed: int,
    exclude_seen_for_negatives: bool,
    negative_sample_count: int,
) -> List[str]:
    if negative_sample_count <= 0:
        raise ValueError(f"negative_sample_count must be > 0, got {negative_sample_count}")

    rng = random.Random(seed)
    negatives: List[str] = []
    used = {chosen_positive}

    for item_id in unit.neg_items:
        if item_id in used:
            continue
        if exclude_seen_for_negatives and item_id in user_seen_item_ids:
            continue
        negatives.append(item_id)
        used.add(item_id)
        if len(negatives) >= negative_sample_count:
            break

    candidate_pool = []
    for item_id in all_item_ids:
        if item_id in used:
            continue
        if exclude_seen_for_negatives and item_id in user_seen_item_ids:
            continue
        candidate_pool.append(item_id)

    rng.shuffle(candidate_pool)
    for item_id in candidate_pool:
        negatives.append(item_id)
        if len(negatives) >= negative_sample_count:
            break

    if len(negatives) < negative_sample_count:
        raise ValueError(
            f"Cannot build {negative_sample_count} negatives for user={unit.user_id}; got {len(negatives)}"
        )

    group = [chosen_positive] + negatives[:negative_sample_count]
    rng.shuffle(group)
    return group


def _bundle_eval_run_root(eval_run_root: str | Path, bundle_output: str | Path) -> Path:
    root = Path(eval_run_root)
    bundle = Path(bundle_output)

    # Allow passing either an explicit zip file path or a directory path.
    if bundle.exists() and bundle.is_dir():
        bundle = bundle / f"{root.name}.zip"
    elif not bundle.exists() and bundle.suffix.lower() != ".zip":
        bundle.mkdir(parents=True, exist_ok=True)
        bundle = bundle / f"{root.name}.zip"

    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in root.rglob("*"):
            if child.is_file():
                zf.write(child, arcname=str(child.relative_to(root)))
    return bundle


def _write_filtered_item_desc(rows: Sequence[Dict[str, str]], keep_item_ids: Set[str], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "image", "summary"], delimiter="\t")
        writer.writeheader()
        for row in rows:
            if row["item_id"] in keep_item_ids:
                writer.writerow(row)


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ------- Metrics (aligned with test_with_llava grouped ranking part) -------
def recall_at_k(y_true: List[List[int]], y_prob: List[List[float]], k: int) -> float:
    recalls = []
    for labels, probs in zip(y_true, y_prob):
        ranked = [x for _, x in sorted(zip(probs, labels), key=lambda t: t[0], reverse=True)]
        retrieved_positives = sum(ranked[:k])
        total_positives = 1  # same assumption as test_with_llava pipeline setting
        recalls.append(retrieved_positives / total_positives)
    return sum(recalls) / len(recalls) if recalls else 0.0


def mrr_at_k(y_true: List[List[int]], y_prob: List[List[float]], k: int) -> float:
    rr = []
    for labels, probs in zip(y_true, y_prob):
        ranked = [x for _, x in sorted(zip(probs, labels), key=lambda t: t[0], reverse=True)]
        val = 0.0
        for i, l in enumerate(ranked[:k]):
            if l == 1:
                val = 1.0 / (i + 1)
                break
        rr.append(val)
    return sum(rr) / len(rr) if rr else 0.0


def ndcg_at_k(y_true: List[List[int]], y_prob: List[List[float]], k: int) -> float:
    out = []
    for labels, probs in zip(y_true, y_prob):
        ranked = [x for _, x in sorted(zip(probs, labels), key=lambda t: t[0], reverse=True)][:k]
        dcg = 0.0
        for i, rel in enumerate(ranked, start=1):
            dcg += (2**rel - 1) / math.log2(i + 1)

        ideal = sorted(labels, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal, start=1):
            idcg += (2**rel - 1) / math.log2(i + 1)

        out.append(dcg / (idcg + 1e-10))
    return sum(out) / len(out) if out else 0.0


def roc_auc_binary(y_true_flat: List[int], y_score_flat: List[float]) -> float:
    # Mann-Whitney U implementation
    n = len(y_true_flat)
    pairs = sorted([(s, y) for s, y in zip(y_score_flat, y_true_flat)], key=lambda x: x[0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            ranks[t] = avg_rank
        i = j + 1

    pos = 0
    neg = 0
    rank_sum_pos = 0.0
    for r, (_, y) in zip(ranks, pairs):
        if y == 1:
            pos += 1
            rank_sum_pos += r
        else:
            neg += 1
    if pos == 0 or neg == 0:
        return 0.0
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _collect_group_scores(
    eval21_items: Sequence[str],
    selected_positive: str,
    ranked_items: Sequence[Dict[str, object]],
) -> Tuple[List[int], List[float]]:
    score_map: Dict[str, float] = {}
    for x in ranked_items:
        iid = str(x.get("item_id", ""))
        s = float(x.get("ranking_score", 0.0))
        if iid:
            score_map[iid] = s

    min_s = min(score_map.values()) if score_map else 0.0
    fallback = min_s - 1.0

    labels: List[int] = []
    scores: List[float] = []
    for iid in eval21_items:
        labels.append(1 if iid == selected_positive else 0)
        scores.append(score_map.get(iid, fallback))
    return labels, scores


def _write_eval21_scored_items(
    out_path: str | Path,
    user_id: str,
    eval21_items: Sequence[str],
    selected_positive: str,
    item_map: Dict[str, Dict[str, str]],
    ranked_items: Sequence[Dict[str, object]],
    labels: Sequence[int],
    scores: Sequence[float],
) -> None:
    ranked_map: Dict[str, Dict[str, object]] = {}
    for rank, x in enumerate(ranked_items, start=1):
        iid = str(x.get("item_id", "")).strip()
        if not iid or iid in ranked_map:
            continue
        ranked_map[iid] = {
            "ranking_score": float(x.get("ranking_score", 0.0)),
            "rank_in_ranked_items": rank,
        }

    records = []
    for iid, label, score in zip(eval21_items, labels, scores):
        base = item_map.get(iid, {})
        ranked_meta = ranked_map.get(iid)
        records.append(
            {
                "user_id": str(user_id),
                "item_id": iid,
                "sample_type": "pos" if int(label) == 1 else "neg",
                "label": int(label),
                "score_for_metrics": float(score),
                "in_ranked_items": bool(ranked_meta is not None),
                "rank_in_ranked_items": int(ranked_meta["rank_in_ranked_items"]) if ranked_meta else None,
                "raw_ranking_score": float(ranked_meta["ranking_score"]) if ranked_meta else None,
                "image": str(base.get("image", "")),
                "summary": str(base.get("summary", "")),
                "is_selected_positive": iid == selected_positive,
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    tsv_path = out.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "user_id",
                "item_id",
                "sample_type",
                "label",
                "score_for_metrics",
                "in_ranked_items",
                "rank_in_ranked_items",
                "raw_ranking_score",
                "image",
                "summary",
                "is_selected_positive",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(records)


def _pick_units(
    units: List[EvalUnit],
    target_user_id: str,
    target_user_row_index: int,
    max_users: int,
    start_user_index: int,
) -> List[EvalUnit]:
    if target_user_id:
        user_units = [u for u in units if u.user_id == target_user_id]
        if not user_units:
            raise ValueError(f"target-user-id={target_user_id} not found")
        idx = max(0, int(target_user_row_index))
        if idx >= len(user_units):
            raise IndexError(f"target-user-row-index={idx} out of range: {len(user_units)}")
        chosen = [user_units[idx]]
    else:
        # keep first row per user to match "每处理完一个user"
        seen: Set[str] = set()
        chosen = []
        for u in units:
            if u.user_id in seen:
                continue
            seen.add(u.user_id)
            chosen.append(u)

    start = max(0, int(start_user_index))
    if start > 0:
        if start >= len(chosen):
            raise IndexError(f"start-user-index={start} out of range: {len(chosen)}")
        chosen = chosen[start:]

    if max_users > 0:
        chosen = chosen[:max_users]
    return chosen


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-user eval21 runner for Amazon full agents pipeline")

    parser.add_argument("--item-desc-tsv", default="./processed/Video_Games_item_desc.tsv")
    parser.add_argument("--user-pairs-tsv", default="./processed/Video_Games_u_i_pairs.tsv")
    parser.add_argument("--eval-user-items-negs-tsv", default="./processed/Video_Games_user_items_negs_test.csv")
    parser.add_argument("--agent2-user-items-negs-tsv", default="./processed/Video_Games_user_items_negs.tsv")
    parser.add_argument(
        "--agent2-item-desc-tsv",
        default="",
        help="Optional full item_desc tsv for Agent2 metadata fallback; defaults to --item-desc-tsv",
    )

    parser.add_argument("--target-user-id", default="")
    parser.add_argument("--target-user-row-index", type=int, default=0)
    parser.add_argument("--positive-index", type=int, default=0)
    parser.add_argument(
        "--positive-selection",
        choices=["latest", "index"],
        default="latest",
        help="How to choose the 1 positive sample from eval row positives.",
    )
    parser.add_argument("--max-users", type=int, default=0, help="0 means all selected users")
    parser.add_argument("--start-user-index", type=int, default=0, help="Skip first N selected users")
    parser.add_argument("--exclude-seen-for-negatives", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--negative-sample-count",
        type=int,
        default=20,
        help="Number of random negative samples to build with the 1 positive item.",
    )
    parser.add_argument(
        "--disable-agent3-item-type-filter",
        action="store_true",
        help=(
            "Disable Agent3 preliminary recall by item type/category. "
            "When enabled, Agent3 sends the whole current eval21 user catalog (20neg+1pos) to Agent5."
        ),
    )

    parser.add_argument("--eval-run-root", default="./processed/eval21_runs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bundle-output", default="./processed/eval21_runs/final_bundle.zip")
    parser.add_argument(
        "--shared-global-db-path",
        default="",
        help="Optional shared global_item_features.db path reused across all users",
    )
    parser.add_argument(
        "--global-db-path",
        default="",
        help="Alias of --shared-global-db-path for passing one reusable global DB path",
    )
    parser.add_argument(
        "--shared-history-db-path",
        default="",
        help="Optional shared user_history_log.db path reused across all users",
    )

    full_defaults = vars(build_full_argparser().parse_args(["--bundle-output", "dummy.zip"]))
    forward_keys = [
        "vl_model",
        "text_model",
        "category_hint",
        "query",
        "min_candidate_items",
        "max_candidate_items",
        "max_history_rows",
        "top_n",
    ]
    for key in forward_keys:
        parser.add_argument(f"--{key.replace('_', '-')}", default=full_defaults[key])
    parser.add_argument(
        "--positive-history-only",
        action="store_true",
        default=bool(full_defaults["positive_history_only"]),
        help="Use only positive history rows in Agent2/Agent3 and ignore Must_Avoid constraints in Agent5.",
    )
    parser.add_argument(
        "--disable-must-have",
        action="store_true",
        default=bool(full_defaults.get("disable_must_have", False)),
        help="Disable Agent4 Must_Have constraints before Agent5 scoring.",
    )
    parser.add_argument(
        "--disable-prediction-bonus",
        action="store_true",
        default=bool(full_defaults.get("disable_prediction_bonus", False)),
        help="Disable Agent5 prediction bonus and use logits-weighted score only.",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    item_rows = _read_item_desc_rows(args.item_desc_tsv)
    item_map = {r["item_id"]: r for r in item_rows}
    all_item_ids = list(item_map.keys())
    required_total_items = int(args.negative_sample_count) + 1
    if len(all_item_ids) < required_total_items:
        raise ValueError(f"item-desc-tsv has <{required_total_items} items")

    units_all = _read_user_items_negs(args.eval_user_items_negs_tsv)
    if not units_all:
        raise ValueError("No valid eval rows")
    units = _pick_units(
        units_all,
        str(args.target_user_id),
        int(args.target_user_row_index),
        int(args.max_users),
        int(args.start_user_index),
    )

    agent2_units_all = _read_user_items_negs(args.agent2_user_items_negs_tsv)
    if not agent2_units_all:
        raise ValueError("No valid rows in agent2-user-items-negs-tsv")

    root = Path(args.eval_run_root)
    root.mkdir(parents=True, exist_ok=True)

    agent2_item_desc_tsv = str(args.agent2_item_desc_tsv or args.item_desc_tsv)
    shared_global_db_path = str(args.shared_global_db_path or args.global_db_path or "").strip()
    shared_history_db_path = str(args.shared_history_db_path or "").strip()
    if shared_global_db_path:
        Path(shared_global_db_path).parent.mkdir(parents=True, exist_ok=True)
    if shared_history_db_path:
        Path(shared_history_db_path).parent.mkdir(parents=True, exist_ok=True)

    grouped_labels: List[List[int]] = []
    grouped_scores: List[List[float]] = []

    total = len(units)
    for idx, unit in enumerate(units, start=1):
        if str(args.positive_selection) == "latest":
            selected_positive = _latest_positive_by_timestamp(
                user_pairs_tsv=args.user_pairs_tsv,
                user_id=unit.user_id,
                positive_item_ids=unit.pos_items,
            )
        else:
            if args.positive_index < 0 or args.positive_index >= len(unit.pos_items):
                raise IndexError(
                    f"positive-index={args.positive_index} out of range for user={unit.user_id}, pos size={len(unit.pos_items)}"
                )
            selected_positive = unit.pos_items[args.positive_index]
        if selected_positive not in item_map:
            raise KeyError(f"positive item {selected_positive} missing in item-desc-tsv")

        print(
            f"[Eval21][Input Progress] user {idx}/{total} {_progress_bar(idx, total)} "
            f"(user_id={unit.user_id}) sample 1/1 {_progress_bar(1, 1)} stage=input"
        )

        user_seen = _user_seen_items(args.user_pairs_tsv, unit.user_id)
        eval21_items = _build_eval21_catalog(
            all_item_ids=all_item_ids,
            unit=unit,
            chosen_positive=selected_positive,
            user_seen_item_ids=user_seen,
            seed=int(args.seed) + idx,
            exclude_seen_for_negatives=bool(args.exclude_seen_for_negatives),
            negative_sample_count=int(args.negative_sample_count),
        )

        user_dir = root / f"user_{unit.user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)

        prepared_item_desc = user_dir / "eval21_item_desc.tsv"
        _write_filtered_item_desc(item_rows, set(eval21_items), prepared_item_desc)

        per_user_agent2_user_items_negs = user_dir / "agent2_user_items_negs.tsv"
        selected_rows = _write_user_items_negs_for_single_user(
            rows=agent2_units_all,
            user_id=unit.user_id,
            out_path=per_user_agent2_user_items_negs,
        )

        meta = {
            "user_id": unit.user_id,
            "selected_positive_item": selected_positive,
            "eval21_items": eval21_items,
            "selected_group_size": len(eval21_items),
            "negative_sample_count": int(args.negative_sample_count),
            "agent2_user_items_negs_rows": selected_rows,
            "agent2_user_items_negs_tsv": str(per_user_agent2_user_items_negs),
            "global_db_path": shared_global_db_path or str(user_dir / "global_item_features.db"),
            "history_db_path": shared_history_db_path or str(user_dir / "user_history_log.db"),
        }
        (user_dir / "eval21_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.prepare_only:
            print(f"[Progress] {idx}/{total} prepared user={unit.user_id}, group_size={len(eval21_items)}")
            continue

        run_args = SimpleNamespace(
            item_desc_tsv=str(prepared_item_desc),
            user_pairs_tsv=str(args.user_pairs_tsv),
            user_items_negs_tsv=str(per_user_agent2_user_items_negs),
            agent2_item_desc_tsv=str(agent2_item_desc_tsv),
            global_db=shared_global_db_path or str(user_dir / "global_item_features.db"),
            history_db=shared_history_db_path or str(user_dir / "user_history_log.db"),
            profiler_run_out_dir=str(user_dir / "profiler_runs"),
            intent_output_dir=str(user_dir / "intent_dual_recall_outputs"),
            dynamic_output_dir=str(user_dir / "dynamic_reasoning_ranking_outputs"),
            bundle_output=str(user_dir / "bundle.zip"),
            vl_model=str(args.vl_model),
            text_model=str(args.text_model),
            category_hint=str(args.category_hint),
            query=str(args.query),
            min_candidate_items=int(args.min_candidate_items),
            max_candidate_items=int(args.max_candidate_items),
            max_history_rows=int(args.max_history_rows),
            top_n=int(args.top_n),
            positive_history_only=bool(args.positive_history_only),
            disable_must_have=bool(args.disable_must_have),
            disable_prediction_bonus=bool(args.disable_prediction_bonus),
            filter_candidates_by_item_type=not bool(args.disable_agent3_item_type_filter),
            candidate_item_ids_scope=list(eval21_items),
        )

        run_pipeline(run_args)
        print(
            f"[Eval21][Output Progress] user {idx}/{total} {_progress_bar(idx, total)} "
            f"(user_id={unit.user_id}) sample 1/1 {_progress_bar(1, 1)} stage=output"
        )

        dyn_path = Path(run_args.dynamic_output_dir) / f"user_{unit.user_id}_dynamic_reasoning_ranking_output.json"
        if not dyn_path.exists():
            raise FileNotFoundError(f"Dynamic output not found: {dyn_path}")
        payload = json.loads(dyn_path.read_text(encoding="utf-8"))
        ranked_items = list(payload.get("ranked_items", []))

        labels, scores = _collect_group_scores(eval21_items, selected_positive, ranked_items)
        _write_eval21_scored_items(
            out_path=user_dir / "eval21_scored_items.json",
            user_id=unit.user_id,
            eval21_items=eval21_items,
            selected_positive=selected_positive,
            item_map=item_map,
            ranked_items=ranked_items,
            labels=labels,
            scores=scores,
        )
        grouped_labels.append(labels)
        grouped_scores.append(scores)

        y_true_flat = [x for row in grouped_labels for x in row]
        y_score_flat = [x for row in grouped_scores for x in row]

        auc = roc_auc_binary(y_true_flat, y_score_flat)
        r3 = recall_at_k(grouped_labels, grouped_scores, 3)
        r5 = recall_at_k(grouped_labels, grouped_scores, 5)
        r10 = recall_at_k(grouped_labels, grouped_scores, 10)
        m3 = mrr_at_k(grouped_labels, grouped_scores, 3)
        m5 = mrr_at_k(grouped_labels, grouped_scores, 5)
        m10 = mrr_at_k(grouped_labels, grouped_scores, 10)
        n3 = ndcg_at_k(grouped_labels, grouped_scores, 3)
        n5 = ndcg_at_k(grouped_labels, grouped_scores, 5)
        n10 = ndcg_at_k(grouped_labels, grouped_scores, 10)

        print(
            f"[Progress] {idx}/{total} user={unit.user_id} | "
            f"AUC={auc:.6f} Recall@3/5/10={r3:.6f}/{r5:.6f}/{r10:.6f} "
            f"MRR@3/5/10={m3:.6f}/{m5:.6f}/{m10:.6f} "
            f"NDCG@3/5/10={n3:.6f}/{n5:.6f}/{n10:.6f}"
        )

    if args.prepare_only:
        bundle_file = _bundle_eval_run_root(root, args.bundle_output)
        print(json.dumps({"prepared_users": total, "eval_run_root": str(root)}, ensure_ascii=False, indent=2))
        print(json.dumps({"bundle_output": str(bundle_file)}, ensure_ascii=False, indent=2))
        return

    final = {
        "processed_users": len(grouped_labels),
        "AUC": roc_auc_binary([x for r in grouped_labels for x in r], [x for r in grouped_scores for x in r]) if grouped_labels else 0.0,
        "Recall@3": recall_at_k(grouped_labels, grouped_scores, 3) if grouped_labels else 0.0,
        "Recall@5": recall_at_k(grouped_labels, grouped_scores, 5) if grouped_labels else 0.0,
        "Recall@10": recall_at_k(grouped_labels, grouped_scores, 10) if grouped_labels else 0.0,
        "MRR@3": mrr_at_k(grouped_labels, grouped_scores, 3) if grouped_labels else 0.0,
        "MRR@5": mrr_at_k(grouped_labels, grouped_scores, 5) if grouped_labels else 0.0,
        "MRR@10": mrr_at_k(grouped_labels, grouped_scores, 10) if grouped_labels else 0.0,
        "NDCG@3": ndcg_at_k(grouped_labels, grouped_scores, 3) if grouped_labels else 0.0,
        "NDCG@5": ndcg_at_k(grouped_labels, grouped_scores, 5) if grouped_labels else 0.0,
        "NDCG@10": ndcg_at_k(grouped_labels, grouped_scores, 10) if grouped_labels else 0.0,
        "eval_run_root": str(root),
    }
    (root / "metrics_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle_file = _bundle_eval_run_root(root, args.bundle_output)
    final["bundle_output"] = str(bundle_file)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(build_argparser().parse_args())
