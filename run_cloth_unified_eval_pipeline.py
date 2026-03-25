from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from image_prefetch import prefetch_item_images

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
except ModuleNotFoundError:
    try:
        from qwen3_vl_embedding import Qwen3VLEmbedder
    except ModuleNotFoundError:
        Qwen3VLEmbedder = None

try:
    from dynamic_reasoning_ranking_agent import run_module3
    from item_profiler_agents import (
        GlobalItemDB,
        HistoryItemProfileInput,
        ItemProfileInput,
        Qwen3VLExtractor,
        UserHistoryLogDB,
    )
    from intent_dual_recall_agent import Qwen3RouterLLM
except ModuleNotFoundError:
    from new_pipe.dynamic_reasoning_ranking_agent import run_module3
    from new_pipe.item_profiler_agents import (
        GlobalItemDB,
        HistoryItemProfileInput,
        ItemProfileInput,
        Qwen3VLExtractor,
        UserHistoryLogDB,
    )
    from new_pipe.intent_dual_recall_agent import Qwen3RouterLLM

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "for", "with", "of", "in", "on", "at", "from", "by",
    "is", "are", "be", "am", "i", "me", "my", "you", "your", "we", "our", "this", "that", "it",
    "want", "need", "looking", "interested", "find", "recommend", "please", "can", "could", "would",
}


def _parse_meta_line(line: str) -> dict:
    t = line.strip()
    if not t:
        return {}
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return ast.literal_eval(t)


def load_filtered_meta(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _parse_meta_line(line)
            asin = str(rec.get("asin", "")).strip()
            if asin:
                out[asin] = rec
    return out


def _meta_category_paths(meta: Dict[str, Any]) -> List[List[str]]:
    categories = meta.get("categories", [])
    out: List[List[str]] = []
    if isinstance(categories, list):
        for cat_path in categories:
            if isinstance(cat_path, list):
                segs = [str(x).strip() for x in cat_path if str(x).strip()]
                if segs:
                    out.append(segs)
    return out


def _meta_category_text(meta: Dict[str, Any]) -> str:
    return " | ".join(" > ".join(x) for x in _meta_category_paths(meta))


def _item_sentence(meta: Dict[str, Any]) -> str:
    return (
        f"categories: {_meta_category_text(meta)}; "
        f"title: {str(meta.get('title', '') or '')}; "
        f"description: {str(meta.get('description', '') or '')}"
    ).strip()


def _query_sentence(
    query: str,
    selected_categories: List[List[str]] | str | None = None,
    rewritten: str | None = None,
) -> str:
    if rewritten is None and isinstance(selected_categories, str):
        rewritten = selected_categories
        selected_categories = []

    normalized_categories = selected_categories if isinstance(selected_categories, list) else []
    cats = " | ".join(" > ".join(seg for seg in c if seg) for c in normalized_categories)
    return f"categories: {cats}; user_need: {rewritten or query}".strip()


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _route_query(query: str, category_catalog: List[str], enable_llm: bool, text_model: str) -> Dict[str, Any]:
    if not enable_llm:
        return {
            "selected_category_paths": [],
            "selected_item_types": [],
            "rewritten_query": query.strip(),
            "reasoning": "rule_based_rewrite_only",
        }

    router = Qwen3RouterLLM(model_name=text_model)
    try:
        routing = router.route(query=query, category_catalog=category_catalog, item_type_catalog=[])
        return {
            "selected_category_paths": routing.category_paths,
            "selected_item_types": routing.item_types,
            "rewritten_query": query.strip(),
            "reasoning": routing.reasoning,
        }
    except Exception as exc:
        print(f"[Agent3] LLM routing failed, fallback to rule-based. error={exc}")
        return {
            "selected_category_paths": [],
            "selected_item_types": [],
            "rewritten_query": query.strip(),
            "reasoning": f"llm_route_failed: {exc}",
        }


def _lightweight_profile(meta: Dict[str, Any], item_id: str) -> Dict[str, Any]:
    category_paths = _meta_category_paths(meta)
    return {
        "item_id": item_id,
        "title": str(meta.get("title", "") or ""),
        "taxonomy": {
            "item_type": category_paths[0][-1] if category_paths else "",
            "category_path": category_paths[0] if category_paths else [],
            "confidence": 0.7,
        },
        "text_tags": {"summary": str(meta.get("description", "") or ""), "price": meta.get("price", None)},
        "visual_tags": {},
        "hypotheses": ["lightweight_profile_without_vl"],
        "overall_confidence": 0.7,
    }


def _cleanup_torch_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int, prompt_name: str | None = None) -> np.ndarray:
    if torch is not None:
        with torch.inference_mode():
            return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, convert_to_numpy=True, show_progress_bar=False)
    return model.encode(texts, batch_size=batch_size, prompt_name=prompt_name, convert_to_numpy=True, show_progress_bar=False)


def _resolve_item_image(meta: Dict[str, Any]) -> str:
    for key in ("imUrl", "image", "image_url", "main_image"):
        value = str(meta.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _encode_multimodal_inputs(
    vl_embedder: Any,
    inputs: List[Dict[str, Any]],
    batch_size: int,
) -> np.ndarray:
    all_chunks: List[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        end = min(len(inputs), start + batch_size)
        batch_inputs = inputs[start:end]
        emb = vl_embedder.process(batch_inputs)
        all_chunks.append(emb.detach().to(torch.float32).cpu().numpy())
        _cleanup_torch_cache()
    return np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros((0, 0), dtype=np.float32)


def _build_item_embedding_cache(
    emb_model: SentenceTransformer | None,
    vl_embedder: Any | None,
    use_multimodal_embedding: bool,
    all_item_ids: List[str],
    meta_map: Dict[str, Dict[str, Any]],
    item_sentence_cache: Dict[str, str],
    emb_cache_path: Path,
    embed_batch_size: int,
    chunk_size: int,
    save_every_n: int,
    local_image_map: Dict[str, str] | None = None,
) -> np.ndarray:
    total = len(all_item_ids)
    print(f"[Agent3] rebuilding item embedding cache for {total} items")
    all_emb_chunks: List[np.ndarray] = []
    shard_paths: List[Path] = []
    processed = 0
    shard_idx = 0
    emb_cache_path.parent.mkdir(parents=True, exist_ok=True)

    for old_shard in emb_cache_path.parent.glob(f"{emb_cache_path.stem}.part_*.npy"):
        old_shard.unlink()

    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        chunk_sentences = []
        for iid in all_item_ids[start:end]:
            sentence = item_sentence_cache.get(iid)
            if not sentence:
                sentence = _item_sentence(meta_map[iid])
                item_sentence_cache[iid] = sentence
            chunk_sentences.append(sentence)

        try:
            if use_multimodal_embedding:
                chunk_inputs = []
                for idx, iid in enumerate(all_item_ids[start:end]):
                    item_input = {"text": chunk_sentences[idx]}
                    image = _resolve_item_image(meta_map[iid])
                    if local_image_map and image in local_image_map:
                        image = local_image_map[image]
                    if image:
                        item_input["image"] = image
                    chunk_inputs.append(item_input)
                chunk_emb = _encode_multimodal_inputs(vl_embedder, chunk_inputs, embed_batch_size)
            else:
                chunk_emb = _encode_texts(emb_model, chunk_sentences, embed_batch_size, prompt_name=None)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and embed_batch_size > 1:
                new_bs = max(1, embed_batch_size // 2)
                print(f"[Agent3] OOM at batch_size={embed_batch_size}, retry {start}-{end} with batch_size={new_bs}")
                if use_multimodal_embedding:
                    chunk_emb = _encode_multimodal_inputs(vl_embedder, chunk_inputs, new_bs)
                else:
                    chunk_emb = _encode_texts(emb_model, chunk_sentences, new_bs, prompt_name=None)
            else:
                raise

        all_emb_chunks.append(chunk_emb.astype(np.float32, copy=False))
        processed = end
        print(f"[Agent3][embedding chunk] {processed}/{total} (chunk={start}-{end})")

        if processed % save_every_n == 0 or processed == total:
            partial = np.concatenate(all_emb_chunks, axis=0)
            shard_path = emb_cache_path.parent / f"{emb_cache_path.stem}.part_{shard_idx:05d}.npy"
            np.save(shard_path, partial)
            shard_paths.append(shard_path)
            all_emb_chunks = []
            shard_idx += 1
            print(f"[Agent3][cache save] {processed}/{total} -> {shard_path}")

        _cleanup_torch_cache()

    if all_emb_chunks:
        shard_path = emb_cache_path.parent / f"{emb_cache_path.stem}.part_{shard_idx:05d}.npy"
        np.save(shard_path, np.concatenate(all_emb_chunks, axis=0))
        shard_paths.append(shard_path)
        all_emb_chunks = []

    if shard_paths:
        final_emb = np.concatenate([np.load(path) for path in shard_paths], axis=0)
    else:
        final_emb = np.zeros((0, 0), dtype=np.float32)
    np.savez_compressed(emb_cache_path, item_ids=np.array(all_item_ids), item_embeddings=final_emb)
    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()
    return final_emb


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _extract_query_keywords(query: str, max_keywords: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query.lower())
    uniq: List[str] = []
    seen = set()
    for t in tokens:
        if t in EN_STOPWORDS:
            continue
        if len(t) <= 1:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= max_keywords:
            break
    return uniq


def _keyword_match_score(title_lower: str, keywords: List[str]) -> Tuple[int, List[str]]:
    matched = [kw for kw in keywords if kw in title_lower]
    return len(matched), matched


def _build_hybrid_recall_ids(
    all_item_ids: List[str],
    title_lower_map: Dict[str, str],
    keywords: List[str],
    rank_indices: np.ndarray,
    keyword_recall_topk: int,
    embedding_recall_topk: int,
) -> Tuple[List[str], int, Dict[str, Any]]:
    keyword_topk = max(1, int(keyword_recall_topk))
    embedding_topk = max(1, int(embedding_recall_topk))

    matched_scored: List[Tuple[int, str, List[str]]] = []
    for iid in all_item_ids:
        score, matched = _keyword_match_score(title_lower_map.get(iid, ""), keywords)
        if score > 0:
            matched_scored.append((score, iid, matched))
    matched_scored.sort(key=lambda x: (-x[0], x[1]))
    matched_ids = [x[1] for x in matched_scored[:keyword_topk]]

    embedding_ids: List[str] = []
    for idx in rank_indices[:embedding_topk]:
        embedding_ids.append(all_item_ids[int(idx)])

    merged_ids: List[str] = []
    seen = set()
    for iid in matched_ids + embedding_ids:
        if iid in seen:
            continue
        merged_ids.append(iid)
        seen.add(iid)

    debug = {
        "keywords": keywords,
        "keyword_matched_count": len(matched_scored),
        "keyword_stage": f"top{keyword_topk}",
        "keyword_pool_size": len(matched_ids),
        "embedding_pool_size": len(embedding_ids),
        "merged_pool_size": len(merged_ids),
        "keyword_recall_topk": keyword_topk,
        "embedding_recall_topk": embedding_topk,
    }
    return merged_ids, len(merged_ids), debug


def _build_history_embedding_recall_ids(
    history_item_ids: List[str],
    item_id_to_index: Dict[str, int],
    item_emb_norm: np.ndarray,
    q_emb_norm: np.ndarray,
    topk: int,
) -> List[str]:
    valid_history_ids = [iid for iid in history_item_ids if iid in item_id_to_index]
    if not valid_history_ids:
        return []

    k = max(1, int(topk))
    hist_indices = np.array([item_id_to_index[iid] for iid in valid_history_ids], dtype=np.int64)
    hist_emb = item_emb_norm[hist_indices]
    sim = np.matmul(hist_emb, q_emb_norm[0])
    order = np.argsort(-sim)
    return [valid_history_ids[int(i)] for i in order[:k]]



def _recall_at_k(labels: List[int], k: int) -> float:
    if not labels:
        return 0.0
    return float(sum(labels[:k]))


def _mrr_at_k(labels: List[int], k: int) -> float:
    for i, label in enumerate(labels[:k], start=1):
        if int(label) == 1:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(labels: List[int], k: int) -> float:
    ranked = labels[:k]
    dcg = 0.0
    for i, rel in enumerate(ranked, start=1):
        dcg += (2 ** int(rel) - 1) / math.log2(i + 1)

    ideal = sorted((int(x) for x in labels), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _safe_item_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("item_id", "")).strip()
    return str(value or "").strip()


def _calc_metrics_from_dynamic_output(path: Path, top_n: int) -> Dict[str, float] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Metrics] skip unreadable file: {path.name} error={exc}")
        return None

    target_id = _safe_item_id(payload.get("groundtruth_target_item_id"))
    if not target_id:
        print(f"[Metrics] skip file without groundtruth_target_item_id: {path.name}")
        return None

    ranked_items = payload.get("ranked_items", [])
    if not isinstance(ranked_items, list):
        ranked_items = []

    top_ranked_ids = [_safe_item_id(x) for x in ranked_items[:top_n]]
    labels = [1 if iid and iid == target_id else 0 for iid in top_ranked_ids]
    if not labels:
        labels = [0]

    return {
        f"recall@{top_n}": _recall_at_k(labels, top_n),
        f"ndcg@{top_n}": _ndcg_at_k(labels, top_n),
        f"mrr@{top_n}": _mrr_at_k(labels, top_n),
    }


def _print_dynamic_output_metrics(output_dir: str | Path, top_ns: List[int] | Tuple[int, ...] = (10, 20, 40)) -> None:
    pattern = str(Path(output_dir) / "*_dynamic_reasoning_ranking_output.json")
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        print(f"[Metrics] no ranking outputs found in {output_dir}")
        return

    normalized_top_ns = []
    seen_top_ns = set()
    for k in top_ns:
        try:
            top_k = int(k)
        except (TypeError, ValueError):
            continue
        if top_k <= 0 or top_k in seen_top_ns:
            continue
        seen_top_ns.add(top_k)
        normalized_top_ns.append(top_k)

    if not normalized_top_ns:
        print(f"[Metrics] no valid top-k values configured for {output_dir}")
        return

    metrics_by_topk: Dict[int, List[Dict[str, float]]] = {k: [] for k in normalized_top_ns}
    for p in paths:
        for top_k in normalized_top_ns:
            row = _calc_metrics_from_dynamic_output(p, top_n=top_k)
            if row is not None:
                metrics_by_topk[top_k].append(row)

    available_topks = [k for k in normalized_top_ns if metrics_by_topk[k]]
    if not available_topks:
        print(f"[Metrics] no valid ranking outputs with groundtruth target in {output_dir}")
        return

    files_count = len(metrics_by_topk[available_topks[0]])
    metric_chunks = []
    for top_k in available_topks:
        metric_rows = metrics_by_topk[top_k]
        recall_key = f"recall@{top_k}"
        ndcg_key = f"ndcg@{top_k}"
        mrr_key = f"mrr@{top_k}"
        recall = float(np.mean([x[recall_key] for x in metric_rows]))
        ndcg = float(np.mean([x[ndcg_key] for x in metric_rows]))
        mrr = float(np.mean([x[mrr_key] for x in metric_rows]))
        metric_chunks.append(
            f"@{top_k} HitRate/Recall={recall:.6f} NDCG={ndcg:.6f} MRR={mrr:.6f}"
        )

    print(f"[Metrics][Aggregated] files={files_count} " + " | ".join(metric_chunks))


def _write_recall_failed_zero_output(output_path: Path, user_id: str, query: str, target_id: str) -> None:
    payload = {
        "user_id": str(user_id),
        "query": str(query),
        "preference_constraints": {
            "Must_Have": [],
            "Nice_to_Have": [],
            "Must_Avoid": [],
            "Predicted_Next_Items": [],
            "Reasoning": "agent3_recall_failed_skip_agent45",
        },
        "ranked_items": [],
        "groundtruth_target_item_id": str(target_id),
        "agent3_recall_hit": 0,
    }
    _save_json(output_path, payload)


def _has_non_empty_ranked_items(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[UserLoop] existing output unreadable, rerun user. path={output_path} error={exc}")
        return False
    ranked_items = payload.get("ranked_items", [])
    return isinstance(ranked_items, list) and len(ranked_items) > 0


def run(args: argparse.Namespace) -> Dict[str, Any]:
    query_df = pd.read_csv(args.query_csv, dtype={"id": str, "user_id": str})
    if args.max_users > 0:
        query_df = query_df.head(args.max_users)

    meta_map = load_filtered_meta(Path(args.filtered_meta_jsonl))
    if not meta_map:
        raise ValueError(f"No items loaded from filtered meta: {args.filtered_meta_jsonl}")

    all_item_ids = sorted(meta_map.keys())
    item_id_to_index = {iid: idx for idx, iid in enumerate(all_item_ids)}
    title_lower_map = {iid: str(meta_map[iid].get("title", "") or "").lower() for iid in all_item_ids}

    cache_dir = Path(args.cache_dir)
    emb_cache_path = cache_dir / "agent3_item_embedding_cache.npz"
    text_cache_path = cache_dir / "agent3_text_cache.json"

    text_cache = _safe_json_load(text_cache_path, {"items": {}, "queries": {}})
    item_sentence_cache: Dict[str, str] = text_cache.get("items", {})
    query_sentence_cache: Dict[str, str] = text_cache.get("queries", {})

    emb_model: SentenceTransformer | None = None
    vl_embedder: Any | None = None
    local_image_map: Dict[str, str] | None = None
    if args.enable_multimodal_embedding:
        if Qwen3VLEmbedder is None:
            raise ImportError("Qwen3VLEmbedder not found. Please ensure qwen3_vl_embedding.py is importable.")
        print(f"[Init] load multimodal embedding model: {args.vl_embedding_model}")
        if args.prefetch_images:
            local_image_map = prefetch_item_images(
                meta_map=meta_map,
                resolve_image_fn=_resolve_item_image,
                cache_dir=Path(args.image_cache_dir),
                max_workers=max(1, int(args.image_prefetch_workers)),
                timeout_sec=max(1, int(args.image_download_timeout_sec)),
            )
        vl_embedder = Qwen3VLEmbedder(model_name_or_path=args.vl_embedding_model)
        emb_cache_path = cache_dir / "agent3_item_embedding_cache_vl.npz"
    else:
        print(f"[Init] load embedding model: {args.embedding_model}")
        emb_model = SentenceTransformer(args.embedding_model)

    item_ids_cached: List[str] = []
    item_emb_matrix: np.ndarray | None = None
    if emb_cache_path.exists():
        npz = np.load(emb_cache_path, allow_pickle=True)
        item_ids_cached = [str(x) for x in npz["item_ids"].tolist()]
        item_emb_matrix = npz["item_embeddings"].astype(np.float32, copy=False)

    if item_emb_matrix is None or item_ids_cached != all_item_ids:
        item_emb_matrix = _build_item_embedding_cache(
            emb_model=emb_model,
            vl_embedder=vl_embedder,
            use_multimodal_embedding=bool(args.enable_multimodal_embedding),
            all_item_ids=all_item_ids,
            meta_map=meta_map,
            item_sentence_cache=item_sentence_cache,
            emb_cache_path=emb_cache_path,
            embed_batch_size=args.embed_batch_size,
            chunk_size=args.embed_chunk_size,
            save_every_n=args.embed_save_every,
            local_image_map=local_image_map,
        )

    item_emb_norm = _l2_normalize(item_emb_matrix)
    global_db = GlobalItemDB(args.global_db)
    history_db = UserHistoryLogDB(args.history_db)
    vl_extractor = Qwen3VLExtractor(model_name=args.vl_model) if args.enable_vl_profiling else None

    category_catalog = sorted({_meta_category_text(v) for v in meta_map.values() if _meta_category_text(v)})
    results: List[Dict[str, Any]] = []

    for row_idx, row in query_df.iterrows():
        user_id = str(row["user_id"])
        target_id = str(row["id"])
        query = str(row.get("new_query") or row.get("query") or "").strip()
        if not query:
            continue

        print(f"\n[UserLoop] {row_idx + 1}/{len(query_df)} user={user_id} target={target_id}")

        existing_output = Path(args.output_dir) / f"user_{user_id}_dynamic_reasoning_ranking_output.json"
        if _has_non_empty_ranked_items(existing_output):
            print(f"[UserLoop] skip user={user_id}: existing non-empty ranking output found at {existing_output}")
            _print_dynamic_output_metrics(args.output_dir)
            continue
        if existing_output.exists():
            print(f"[UserLoop] user={user_id} has empty ranked_items output, retry Agent3 recall before deciding skip")

        routed = _route_query(query, category_catalog, args.enable_llm_routing, args.text_model)

        q_sentence = _query_sentence(query, routed.get("selected_category_paths", []) or [], routed["rewritten_query"])
        query_sentence_cache[f"{user_id}::{q_sentence}"] = q_sentence
        history_ids = [x for x in str(row.get("remaining_interaction_string", "")).split("|") if x]

        filtered_item_ids = all_item_ids
        print(f"[Agent3][categories] skip exact-match prefilter; candidate_count={len(filtered_item_ids)}")

        filtered_idx = [item_id_to_index[iid] for iid in filtered_item_ids]
        filtered_emb = item_emb_norm[np.array(filtered_idx)]

        if args.enable_multimodal_embedding:
            q_emb = vl_embedder.process([{"text": q_sentence}]).detach().to(torch.float32).cpu().numpy()
        else:
            q_emb = _encode_texts(emb_model, [q_sentence], batch_size=1, prompt_name="query").astype(np.float32, copy=False)
        q_emb_norm = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
        sim_matrix = np.matmul(filtered_emb, q_emb_norm[0])
        rank_indices = np.argsort(-sim_matrix)

        keywords = _extract_query_keywords(query, max_keywords=args.max_query_keywords)
        top_ids, used_k, kw_debug = _build_hybrid_recall_ids(
            all_item_ids=filtered_item_ids,
            title_lower_map=title_lower_map,
            keywords=keywords,
            rank_indices=rank_indices,
            keyword_recall_topk=args.keyword_recall_topk or args.fixed_recall_topk,
            embedding_recall_topk=args.embedding_recall_topk or args.fixed_recall_topk,
        )
        history_recall_ids = _build_history_embedding_recall_ids(
            history_item_ids=history_ids,
            item_id_to_index=item_id_to_index,
            item_emb_norm=item_emb_norm,
            q_emb_norm=q_emb_norm,
            topk=args.agent3_history_embedding_topk,
        )
        merged_ids: List[str] = []
        seen_merged = set()
        for iid in top_ids + history_recall_ids:
            if iid in seen_merged:
                continue
            seen_merged.add(iid)
            merged_ids.append(iid)
        top_ids = merged_ids
        used_k = len(top_ids)
        kw_debug["history_embedding_topk"] = int(args.agent3_history_embedding_topk)
        kw_debug["history_embedding_pool_size"] = len(history_recall_ids)
        kw_debug["merged_pool_size"] = len(top_ids)
        print(
            f"[Agent3][keyword] keywords={kw_debug['keywords']} matched={kw_debug['keyword_matched_count']} "
            f"stage={kw_debug['keyword_stage']} prefilter_size={len(filtered_item_ids)}"
        )

        hit = target_id in top_ids
        if not hit:
            print("[Agent3] recall failed. metric=0, skip Agent1/2/4/5")
            _write_recall_failed_zero_output(
                output_path=existing_output,
                user_id=user_id,
                query=q_sentence,
                target_id=target_id,
            )
            results.append({"user_id": user_id, "target_id": target_id, "hit": 0, "used_k": used_k, "kw_debug": kw_debug})
            _print_dynamic_output_metrics(args.output_dir)
            continue

        print(f"[Agent3] recall hit at k={used_k}; run Agent1/2")
        candidate_items: List[Dict[str, Any]] = []
        for i, iid in enumerate(top_ids, start=1):
            meta = meta_map[iid]
            profile = global_db.get_profile(iid)
            if profile is None:
                if vl_extractor is not None:
                    item_input = ItemProfileInput(
                        item_id=iid,
                        title=str(meta.get("title", "") or f"item_{iid}"),
                        detail_text=str(meta.get("description", "") or ""),
                        main_image=str(meta.get("imUrl", "") or ""),
                        detail_images=[],
                        price=str(meta.get("price", "") or ""),
                        category_hint=_meta_category_text(meta),
                    )
                    profile = vl_extractor.extract(prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}", image_paths=[item_input.main_image])
                else:
                    profile = _lightweight_profile(meta, iid)
                global_db.upsert(iid, profile)

            candidate_items.append({"item_id": iid, "profile": profile})
            if i % 50 == 0 or i == len(top_ids):
                print(f"[Agent1] {i}/{len(top_ids)}")

        history_rows: List[Dict[str, Any]] = []
        for i, iid in enumerate(history_ids, start=1):
            meta = meta_map.get(iid)
            if meta is None:
                continue

            profile = global_db.get_profile(iid)
            if profile is None:
                if vl_extractor is not None:
                    item_input = HistoryItemProfileInput(
                        user_id=user_id,
                        item_id=iid,
                        title=str(meta.get("title", "") or f"item_{iid}"),
                        detail_text=str(meta.get("description", "") or ""),
                        main_image=str(meta.get("imUrl", "") or ""),
                        behavior="positive",
                        timestamp=None,
                    )
                    profile = vl_extractor.extract(prompt=f"Profile item: {item_input.title}\n{item_input.detail_text}", image_paths=[item_input.main_image])
                else:
                    profile = _lightweight_profile(meta, iid)
                global_db.upsert(iid, profile)

            if not history_db.exists(user_id=user_id, item_id=iid, behavior="positive", timestamp=None):
                history_db.insert(user_id=user_id, item_id=iid, behavior="positive", timestamp=None, profile=profile)

            history_rows.append({"user_id": user_id, "item_id": iid, "behavior": "positive", "timestamp": None, "profile": profile})
            if i % 20 == 0 or i == len(history_ids):
                print(f"[Agent2] {i}/{len(history_ids)}")

        agent3_output = {
            "query": q_sentence,
            "user_id": user_id,
            "routing": routed,
            "candidate_items": candidate_items,
            "query_relevant_history": history_rows,
        }

        ranked_first = ""
        if args.enable_agent45:
            module3_out = run_module3(
                intent_dual_recall_output=agent3_output,
                model_name=args.text_model,
                top_n=args.top_n,
                save_output=True,
                output_dir=args.output_dir,
                groundtruth_target_item_id=target_id,
                collaborative_db_path=args.collaborative_db_path,
                collaborative_embedding_model=args.collaborative_embedding_model,
                collaborative_similarity_threshold=args.collaborative_similarity_threshold,
                collaborative_top_k=args.collaborative_top_k,
            )
            ranked_first = module3_out.ranked_items[0]["item_id"] if module3_out.ranked_items else ""
        else:
            print("[Agent4/5] skipped by --disable-agent45")

        results.append({
            "user_id": user_id,
            "target_id": target_id,
            "hit": 1,
            "used_k": used_k,
            "top1": ranked_first,
            "kw_debug": kw_debug,
        })
        _print_dynamic_output_metrics(args.output_dir)

    text_cache["items"] = item_sentence_cache
    text_cache["queries"] = query_sentence_cache
    _save_json(text_cache_path, text_cache)
    _save_json(Path(args.output_dir) / "unified_eval_results.json", results)

    recall_rate = float(np.mean([r["hit"] for r in results])) if results else 0.0
    summary = {"rows": len(results), "recall@k": recall_rate, "output_dir": args.output_dir}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可运行的 Beauty 统一评估流程：Agent3 -> Agent1/2 -> Agent4/5")
    parser.add_argument("--query-csv", default="data/amazon_beauty/query_data1.csv")
    parser.add_argument("--filtered-meta-jsonl", default="data/amazon_beauty/meta_Beauty.filtered.jsonl")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--enable-multimodal-embedding", action="store_true", help="开启后，Agent3召回向量改为Qwen3-VL多模态embedding（文本+图片）。")
    parser.add_argument("--vl-embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--embed-chunk-size", type=int, default=20000)
    parser.add_argument("--embed-save-every", type=int, default=1000)
    parser.add_argument("--prefetch-images", action="store_true", help="先集中下载图片到本地缓存，再从本地路径做多模态embedding。")
    parser.add_argument("--image-cache-dir", default="processed/image_cache/cloth")
    parser.add_argument("--image-prefetch-workers", type=int, default=16)
    parser.add_argument("--image-download-timeout-sec", type=int, default=8)
    parser.add_argument("--fixed-recall-topk", type=int, default=250, help="兼容旧参数：当未单独指定关键词/embedding召回Top-K时，作为二者的共同默认值。")
    parser.add_argument("--keyword-recall-topk", type=int, default=0, help="Agent3基于标题关键词匹配的召回Top-K；<=0时回退到--fixed-recall-topk。")
    parser.add_argument("--embedding-recall-topk", type=int, default=0, help="Agent3基于embedding相似度的召回Top-K；<=0时回退到--fixed-recall-topk。")
    parser.add_argument("--agent3-history-embedding-topk", type=int, default=20, help="Agent3额外从用户历史中按query向量相似度召回Top-K（不足则全召回）。")
    parser.add_argument("--max-query-keywords", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--max-users", type=int, default=0, help="仅跑前N条query，0表示全量")

    parser.add_argument("--cache-dir", default="processed/beauty_cache")
    parser.add_argument("--output-dir", default="processed/beauty_unified_outputs")
    parser.add_argument("--global-db", default="processed/beauty_global_item_features.db")
    parser.add_argument("--history-db", default="processed/beauty_user_history.db")

    parser.add_argument("--vl-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--enable-llm-routing", action="store_true", help="开启Qwen3文本路由；默认关闭走规则fallback")
    parser.add_argument("--enable-vl-profiling", action="store_true", help="开启Qwen3-VL画像；默认关闭走轻量画像")
    parser.add_argument("--disable-agent45", action="store_true", help="关闭Agent4/5")
    parser.add_argument("--collaborative-db-path", default="processed/collaborative_preference_memory.db")
    parser.add_argument("--collaborative-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--collaborative-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--collaborative-top-k", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.enable_agent45 = not bool(args.disable_agent45)
    run(args)
