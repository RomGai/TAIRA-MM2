"""MLLM-based item profiling agents for Amazon processed data.

This module builds two profiling agents around the output format from
`process_data.py`:

- Agent 1 (CandidateItemProfiler): profile candidate items and write
  structured features into a global item DB.
- Agent 2 (HistoryItemProfiler): profile historical interacted items and write
  item features with behavior label + timestamp into a user history log DB.

The implementation is designed for Qwen3VL-8B style multimodal models.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

try:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover - allow lightweight environments
    torch = None
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None


BehaviorLabel = Literal["positive", "negative"]


@dataclass
class ItemProfileInput:
    item_id: str
    title: str
    detail_text: str
    main_image: str
    detail_images: List[str] = field(default_factory=list)
    price: Optional[str] = None
    brand: Optional[str] = None
    category_hint: Optional[str] = None


@dataclass
class HistoryItemProfileInput(ItemProfileInput):
    user_id: str = ""
    behavior: BehaviorLabel = "positive"
    timestamp: Optional[int] = 0


def _normalize_timestamp_for_db(ts: Optional[int]) -> int:
    """Normalize optional timestamp to DB-safe integer.

    Negative samples may have no real interaction timestamp. We store such cases as -1.
    """
    if ts is None:
        return -1
    return int(ts)


class Qwen3VLExtractor:
    """Qwen3-VL wrapper following official transformers usage style."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "auto",
        max_new_tokens: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or int(os.getenv("out_seq_length", "16384"))
        self.device = device
        self.torch_dtype = torch_dtype
        self.do_sample = os.getenv("greedy", "false").lower() != "true"
        self.top_p = float(os.getenv("top_p", "0.8"))
        self.top_k = int(os.getenv("top_k", "20"))
        self.temperature = float(os.getenv("temperature", "0.7"))
        self.repetition_penalty = float(os.getenv("repetition_penalty", "1.0"))
        self.presence_penalty = float(os.getenv("presence_penalty", "1.5"))
        self.json_retry = int(os.getenv("json_retry", "1"))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if AutoProcessor is None or Qwen3VLForConditionalGeneration is None or torch is None:
            raise ImportError(
                "transformers/torch are not available. Install required dependencies first."
            )

        if self._model is not None and self._processor is not None:
            return

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        if self.torch_dtype == "auto":
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
            )
        else:
            dtype = getattr(torch, self.torch_dtype)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

        if self.device != "cuda":
            self._model.to(self.device)

    def _generate_text(self, messages: List[Dict[str, Any]], force_greedy: bool = False) -> str:
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False if force_greedy else self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "temperature": 0.0 if force_greedy else self.temperature,
            "repetition_penalty": self.repetition_penalty,
#            "presence_penalty": self.presence_penalty,
        }
        if force_greedy:
            generate_kwargs.pop("top_p", None)
            generate_kwargs.pop("top_k", None)

        try:
            output_ids = self._model.generate(**inputs, **generate_kwargs)
        except TypeError:
            # Some transformers versions do not support `presence_penalty` in generate().
            generate_kwargs.pop("presence_penalty", None)
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return generated_text

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
        decoder = json.JSONDecoder()
        stripped = text.strip()

        # 1) direct decode
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        # 2) markdown json code fence
        if "```" in stripped:
            parts = stripped.split("```")
            for part in parts:
                candidate = part.replace("json", "", 1).strip()
                if not candidate:
                    continue
                try:
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue

        # 3) find first decodable JSON object from any '{' start
        for i, ch in enumerate(stripped):
            if ch != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(stripped, i)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        return None



    @staticmethod
    def _normalize_image_paths(image_paths: List[str]) -> List[str]:
        """Drop empty/obviously invalid image entries before processor ingestion."""
        cleaned: List[str] = []
        for p in image_paths:
            cand = str(p or "").strip()
            if not cand:
                continue
            # Avoid placeholders that are known to break image loading.
            if cand in {".", "./", "..", "../"}:
                continue
            cleaned.append(cand)
        return cleaned

    def extract(
        self,
        prompt: str,
        image_paths: List[str],
    ) -> Dict[str, Any]:
        self.load()

        valid_image_paths = self._normalize_image_paths(image_paths)
        image_messages = [{"type": "image", "image": path} for path in valid_image_paths]
        messages = [
            {
                "role": "user",
                "content": [
                    *image_messages,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            generated_text = self._generate_text(messages)
        except Exception as exc:
            # Some rows contain invalid image URLs/paths; fallback to text-only profiling.
            print(f"[Qwen3VLExtractor] image loading failed, fallback to text-only: {exc}")
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            generated_text = self._generate_text(messages)

        parsed = self._try_json_decode(generated_text)
        if parsed is not None:
            return parsed

        # Retry with stricter formatting instruction to reduce JSON parsing failures.
        for retry_idx in range(self.json_retry):
            strict_messages = [
                {
                    "role": "user",
                    "content": [
                        *image_messages,
                        {
                            "type": "text",
                            "text": (
                                prompt
                                + "\n\nIMPORTANT: Output exactly one valid JSON object only. "
                                + "Do not include markdown/code fences/comments/trailing text."
                            ),
                        },
                    ],
                }
            ]
            if not image_messages:
                strict_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    prompt
                                    + "\n\nIMPORTANT: Output exactly one valid JSON object only. "
                                    + "Do not include markdown/code fences/comments/trailing text."
                                ),
                            }
                        ],
                    }
                ]
            generated_text = self._generate_text(strict_messages, force_greedy=True)
            parsed = self._try_json_decode(generated_text)
            if parsed is not None:
                return parsed

        raise ValueError(
            "Model output is not valid JSON after retries. "
            f"Last output (truncated): {generated_text[:2000]}"
        )



class GlobalItemDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_item_features (
                item_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, item_id: str, profile: Dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO global_item_features (item_id, profile_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                profile_json=excluded.profile_json,
                updated_at=excluded.updated_at
            """,
            (
                item_id,
                json.dumps(profile, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_profile(self, item_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT profile_json FROM global_item_features WHERE item_id = ?",
            (str(item_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])


class UserHistoryLogDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_history_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                behavior TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_time
            ON user_history_profiles (user_id, timestamp)
            """
        )
        self.conn.commit()

    def insert(
        self,
        user_id: str,
        item_id: str,
        behavior: BehaviorLabel,
        timestamp: Optional[int],
        profile: Dict[str, Any],
    ) -> None:
        timestamp_db = _normalize_timestamp_for_db(timestamp)
        self.conn.execute(
            """
            INSERT INTO user_history_profiles
            (user_id, item_id, behavior, timestamp, profile_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                item_id,
                behavior,
                timestamp_db,
                json.dumps(profile, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def exists(
        self,
        user_id: str,
        item_id: str,
        behavior: BehaviorLabel,
        timestamp: Optional[int],
    ) -> bool:
        timestamp_db = _normalize_timestamp_for_db(timestamp)
        cursor = self.conn.execute(
            """
            SELECT 1 FROM user_history_profiles
            WHERE user_id = ? AND item_id = ? AND behavior = ? AND timestamp = ?
            LIMIT 1
            """,
            (str(user_id), str(item_id), str(behavior), timestamp_db),
        )
        return cursor.fetchone() is not None


def build_profile_prompt(item: ItemProfileInput) -> str:
    """Prompt template for fine-grained textual + visual profiling."""

    image_count = 1 + len(item.detail_images)
    return f"""
You are an expert e-commerce item profiler.
Given product text and {image_count} images (first is main image, rest are detail images),
extract a fine-grained feature profile in STRICT JSON.

Item text fields:
- title: {item.title}
- detail_text: {item.detail_text}
- brand: {item.brand or ''}
- price: {item.price or ''}
- category_hint: {item.category_hint or ''}

User shopping-oriented extraction requirements:
1) Type-first taxonomy (important):
   - Always output `item_type` (required).
   - If hierarchical category is uncertain, keep only `item_type` and leave `category_path` empty.
   - If known, output `category_path` as a list (e.g., ["Electronics", "Gaming", "Headset"]).
   - Also infer use_case, target_people, seasonality.
2) Textual attribute tags (fine-grained):
   - title keyword summary (must leverage title)
   - material/fabric composition
   - core features & specs (size, capacity, weight, dimensions, compatibility, power, ingredients)
   - package/bundle information
   - quality & durability claims
   - comfort/usability claims
   - price_band inference (budget/mid/premium) and value_for_money signal
3) Visual style tags (from images):
   - dominant colors (+ optional hex-like names)
   - silhouette/shape/form factor
   - style keywords (minimalist, sporty, retro, luxury, kawaii, etc.)
   - texture/finish (matte/glossy/metallic/knit/grainy)
   - pattern/print/logo density
   - scene mood (homey, professional, outdoor, gaming, etc.)
   - perceived quality level (low/medium/high with confidence)
4) Output quality:
   - Every major field must include a confidence in [0,1].
   - Put uncertain values under "hypotheses".
   - Output ONLY one JSON object. No markdown.

JSON schema:
{{
  "item_id": "{item.item_id}",
  "title": "{item.title}",
  "taxonomy": {{
    "item_type": "",
    "category_path": [],
    "use_case": [],
    "target_people": [],
    "seasonality": "",
    "confidence": 0.0
  }},
  "text_tags": {{...}},
  "visual_tags": {{...}},
  "hypotheses": ["..."],
  "overall_confidence": 0.0
}}
""".strip()


class CandidateItemProfiler:
    """Agent 1: Candidate Item Profiler."""

    def __init__(self, extractor: Qwen3VLExtractor, global_db: GlobalItemDB) -> None:
        self.extractor = extractor
        self.global_db = global_db

    def profile_and_store(self, item: ItemProfileInput) -> Dict[str, Any]:
        prompt = build_profile_prompt(item)
        image_paths = [item.main_image, *item.detail_images]
        profile = self.extractor.extract(prompt, image_paths)
        self.global_db.upsert(item.item_id, profile)
        return profile


class HistoryItemProfiler:
    """Agent 2: History Item Profiler."""

    def __init__(self, extractor: Qwen3VLExtractor, history_db: UserHistoryLogDB) -> None:
        self.extractor = extractor
        self.history_db = history_db

    def profile_and_store(self, item: HistoryItemProfileInput) -> Dict[str, Any]:
        prompt = build_profile_prompt(item)
        image_paths = [item.main_image, *item.detail_images]
        profile = self.extractor.extract(prompt, image_paths)
        profile["behavior"] = item.behavior
        profile["timestamp"] = int(item.timestamp)
        profile["user_id"] = item.user_id

        self.history_db.insert(
            user_id=item.user_id,
            item_id=item.item_id,
            behavior=item.behavior,
            timestamp=item.timestamp,
            profile=profile,
        )
        return profile


def load_item_desc_tsv(path: str | Path) -> Dict[str, Dict[str, str]]:
    """Load `*_item_desc.tsv` created by process_data.py into item metadata map."""
    item_map: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item_map[str(row["item_id"])] = {
                "image": row.get("image", ""),
                "summary": row.get("summary", ""),
            }
    return item_map


def load_user_interactions(path: str | Path) -> Iterable[Dict[str, str]]:
    """Load `*_u_i_pairs.tsv` rows."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def expand_pos_neg_rows(
    user_items_negs_path: str | Path,
) -> Iterable[Dict[str, Any]]:
    """Expand `*_user_items_negs.tsv` to (user,item,behavior) rows."""
    with open(user_items_negs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            user = str(row["user_id"])
            for item in str(row["pos"]).split(","):
                if item:
                    yield {"user_id": user, "item_id": item, "behavior": "positive"}
            for item in str(row["neg"]).split(","):
                if item:
                    yield {"user_id": user, "item_id": item, "behavior": "negative"}


def bootstrap_agents_from_processed(
    item_desc_tsv: str | Path,
    global_db_path: str | Path,
    history_db_path: str | Path,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> tuple[CandidateItemProfiler, HistoryItemProfiler]:
    """Build both agents from processed Amazon format."""
    _ = load_item_desc_tsv(item_desc_tsv)  # preload check to ensure input validity

    extractor = Qwen3VLExtractor(model_name=model_name)
    global_db = GlobalItemDB(global_db_path)
    history_db = UserHistoryLogDB(history_db_path)
    return CandidateItemProfiler(extractor, global_db), HistoryItemProfiler(extractor, history_db)


def _sample_distinct_items(
    item_map: Dict[str, Dict[str, str]],
    k: int,
) -> List[str]:
    """Take up to k distinct item_ids in original order from item metadata map."""
    item_ids = list(item_map.keys())
    return item_ids[: min(k, len(item_ids))]


def _sample_distinct_user_item_rows(
    rows: Iterable[Dict[str, str]],
    k: int,
    seed: int = 2025,
) -> List[Dict[str, str]]:
    """Sample up to k rows with distinct user_id and item_id."""
    all_rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(all_rows)

    picked: List[Dict[str, str]] = []
    used_users = set()
    used_items = set()

    for row in all_rows:
        user_id = str(row.get("user_id", ""))
        item_id = str(row.get("item_id", ""))
        if not user_id or not item_id:
            continue
        if user_id in used_users or item_id in used_items:
            continue
        picked.append(row)
        used_users.add(user_id)
        used_items.add(item_id)
        if len(picked) >= k:
            break

    return picked


def _pick_single_user_full_sequence(
    rows: Iterable[Dict[str, str]],
    seed: int = 2025,
) -> List[Dict[str, str]]:
    """Pick one user and return their full interaction sequence sorted by timestamp."""
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        uid = str(row.get("user_id", ""))
        if not uid:
            continue
        grouped.setdefault(uid, []).append(row)

    if not grouped:
        return []

    # Prefer users with longer sequences; break tie randomly but reproducibly.
    users = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(users)
    users.sort(key=lambda u: len(grouped[u]), reverse=True)
    picked_user = users[0]

    seq = grouped[picked_user]
    seq.sort(key=lambda r: int(r.get("timestamp", 0)))
    return seq


def _build_user_item_timestamp_map(user_pairs_tsv_path: str | Path) -> Dict[tuple[str, str], int]:
    """Build (user_id, item_id) -> latest timestamp map from u_i_pairs."""
    ts_map: Dict[tuple[str, str], int] = {}
    for row in load_user_interactions(user_pairs_tsv_path):
        user_id = str(row.get("user_id", ""))
        item_id = str(row.get("item_id", ""))
        ts = int(row.get("timestamp", 0))
        key = (user_id, item_id)
        if key not in ts_map or ts > ts_map[key]:
            ts_map[key] = ts
    return ts_map


def _pick_multi_user_labeled_sequences(
    user_pairs_tsv_path: str | Path,
    user_items_negs_path: str | Path,
    num_users: int = 2,
    max_rows: int = 500,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Pick multiple users from pos/neg labels and return labeled sequences.

    - Positive labels use timestamps joined from `*_u_i_pairs.tsv`.
    - Negative labels are kept even when no timestamp exists (timestamp will be None).
    """
    ts_map = _build_user_item_timestamp_map(user_pairs_tsv_path)
    stats = {
        "pos_rows_in_negs": 0,
        "neg_rows_in_negs": 0,
        "pos_rows_with_timestamp": 0,
        "neg_rows_with_timestamp": 0,
        "rows_dropped_missing_timestamp": 0,
        "neg_rows_without_timestamp_kept": 0,
    }

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in expand_pos_neg_rows(user_items_negs_path):
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        behavior = str(row["behavior"])
        if behavior == "positive":
            stats["pos_rows_in_negs"] += 1
        elif behavior == "negative":
            stats["neg_rows_in_negs"] += 1

        ts = ts_map.get((user_id, item_id))
        if behavior == "positive":
            if ts is None:
                # Positive interactions should have real timestamps from interactions file.
                stats["rows_dropped_missing_timestamp"] += 1
                continue
            stats["pos_rows_with_timestamp"] += 1
        elif behavior == "negative":
            if ts is None:
                stats["neg_rows_without_timestamp_kept"] += 1
            else:
                stats["neg_rows_with_timestamp"] += 1

        grouped.setdefault(user_id, []).append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "behavior": behavior,
                "timestamp": ts,
            }
        )

    if not grouped:
        return [], stats

    # Keep file encounter order (dict insertion order) and pick first N users.
    users = list(grouped.keys())
    selected_users = users[: max(1, num_users)]
    merged: List[Dict[str, Any]] = []
    for uid in selected_users:
        seq = grouped[uid]
        # Order rule:
        # 1) positives first in strict chronological order (smaller/earlier timestamp first)
        # 2) negatives after positives (timestamp not required)
        seq.sort(
            key=lambda r: (
                0 if r["behavior"] == "positive" else 1,
                int(r["timestamp"]) if r["timestamp"] is not None else 10**30,
                str(r["item_id"]),
            )
        )
        merged.extend(seq[:max_rows])

    # Keep per-user timestamp order and selected user order.
    return merged, stats


def _write_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_sqlite_table_as_jsonl(db_path: str | Path, table: str, out_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(f"SELECT * FROM {table}")
        rows = [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
    _write_jsonl(out_path, rows)


if __name__ == "__main__":
    # Real runnable example:
    # - warm up candidate item profiles with batched runs
    # - pick two users and run full labeled history sequence modeling by timestamp order
    # - run both profilers and print profile outputs
    sample_k = int(os.getenv("candidate_sample_k", "150"))
    batch_size = int(os.getenv("batch_size", "4"))
    max_history_rows = int(os.getenv("max_history_rows", "500"))
    item_desc_tsv_path = "./processed/Video_Games_item_desc.tsv"
    user_pairs_tsv_path = "./processed/Video_Games_u_i_pairs.tsv"
    user_items_negs_tsv_path = "./processed/Video_Games_user_items_negs.tsv"
    run_out_dir = Path("./processed/profiler_runs/shared")
    run_out_dir.mkdir(parents=True, exist_ok=True)
    global_db_path = "./processed/global_item_features.db"
    history_db_path = "./processed/user_history_log.db"

    candidate_profiler, history_profiler = bootstrap_agents_from_processed(
        item_desc_tsv=item_desc_tsv_path,
        global_db_path=global_db_path,
        history_db_path=history_db_path,
    )

    item_map = load_item_desc_tsv(item_desc_tsv_path)
    sampled_item_ids = _sample_distinct_items(item_map, k=sample_k)
    candidate_meta_records: List[Dict[str, Any]] = []
    candidate_profile_records: List[Dict[str, Any]] = []

    print(
        f"\n[Agent 1] Batch profiling warm-up on {len(sampled_item_ids)} sampled items "
        f"(batch_size={batch_size}) with incremental reuse..."
    )
    for idx, item_id in enumerate(sampled_item_ids, start=1):
        meta = item_map[item_id]
        sample_item = ItemProfileInput(
            item_id=item_id,
            title=f"item_{item_id}",
            detail_text=meta.get("summary", "") or "",
            main_image=meta.get("image", ""),
            detail_images=[],
            category_hint="Video_Games",
        )
        candidate_meta_records.append(asdict(sample_item))
        profile = candidate_profiler.global_db.get_profile(item_id)
        profile_source = "global_db_reused"
        if profile is None:
            profile = candidate_profiler.profile_and_store(sample_item)
            profile_source = "newly_profiled"
        candidate_profile_records.append(
            {
                "item_id": item_id,
                "source": "candidate",
                "profile_source": profile_source,
                "profile": profile,
            }
        )
        print(
            f"\n[Agent 1][{idx}/{len(sampled_item_ids)}] item_id={item_id}, "
            f"profile_source={profile_source}"
        )
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        if idx % batch_size == 0:
            print(f"[Agent 1] batch progress: {idx}/{len(sampled_item_ids)}")

    sampled_user_rows, label_parse_stats = _pick_multi_user_labeled_sequences(
        user_pairs_tsv_path,
        user_items_negs_tsv_path,
        num_users=3,
        max_rows=max_history_rows,
    )
    history_meta_records: List[Dict[str, Any]] = []
    history_profile_records: List[Dict[str, Any]] = []

    chosen_user_ids = sorted({str(r["user_id"]) for r in sampled_user_rows})
    print("\n[Agent 2] Parsed label/timestamp stats:")
    print(json.dumps(label_parse_stats, ensure_ascii=False, indent=2))
    print(
        "\n[Agent 2] Running full-sequence history profiling "
        f"for user_ids={chosen_user_ids} with {len(sampled_user_rows)} interactions..."
    )
    for idx, row in enumerate(sampled_user_rows, start=1):
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        timestamp_raw = row.get("timestamp")
        timestamp: Optional[int] = None if timestamp_raw is None else int(timestamp_raw)
        meta = item_map.get(item_id, {"image": "", "summary": ""})
        hist_item = HistoryItemProfileInput(
            item_id=item_id,
            title=f"item_{item_id}",
            detail_text=meta.get("summary", "") or "",
            main_image=meta.get("image", ""),
            detail_images=[],
            category_hint="Video_Games",
            user_id=user_id,
            behavior=str(row["behavior"]),
            timestamp=timestamp,
        )
        history_meta_records.append(asdict(hist_item))

        # Reuse existing item profile if already modeled in global item DB.
        profile = candidate_profiler.global_db.get_profile(item_id)
        profile_source = "global_db_reused"
        if profile is None:
            profile = candidate_profiler.profile_and_store(
                ItemProfileInput(
                    item_id=item_id,
                    title=hist_item.title,
                    detail_text=hist_item.detail_text,
                    main_image=hist_item.main_image,
                    detail_images=hist_item.detail_images,
                    category_hint=hist_item.category_hint,
                )
            )
            profile_source = "newly_profiled"

        profile["behavior"] = hist_item.behavior
        profile["timestamp"] = hist_item.timestamp
        profile["user_id"] = hist_item.user_id
        if not history_profiler.history_db.exists(
            user_id=hist_item.user_id,
            item_id=hist_item.item_id,
            behavior=hist_item.behavior,
            timestamp=hist_item.timestamp,
        ):
            history_profiler.history_db.insert(
                user_id=hist_item.user_id,
                item_id=hist_item.item_id,
                behavior=hist_item.behavior,
                timestamp=hist_item.timestamp,
                profile=profile,
            )
            history_write_status = "inserted"
        else:
            history_write_status = "already_exists"
        history_profile_records.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "timestamp": timestamp,
                "source": "history",
                "profile_source": profile_source,
                "history_write_status": history_write_status,
                "profile": profile,
            }
        )
        print(
            f"\n[Agent 2][{idx}/{len(sampled_user_rows)}] "
            f"user_id={user_id}, item_id={item_id}, ts={timestamp}, "
            f"behavior={hist_item.behavior}, profile_source={profile_source}, "
            f"history_write_status={history_write_status}"
        )
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        if idx % batch_size == 0:
            print(f"[Agent 2] batch progress: {idx}/{len(sampled_user_rows)}")

    # Save meta and generated profiles for manual verification.
    _write_jsonl(run_out_dir / "candidate_meta.jsonl", candidate_meta_records)
    _write_jsonl(run_out_dir / "history_meta.jsonl", history_meta_records)
    _write_jsonl(run_out_dir / "candidate_profiles.jsonl", candidate_profile_records)
    _write_jsonl(run_out_dir / "history_profiles.jsonl", history_profile_records)

    # Snapshot DB tables into jsonl for easy manual diff/check.
    _export_sqlite_table_as_jsonl(
        global_db_path,
        "global_item_features",
        run_out_dir / "global_item_features_snapshot.jsonl",
    )
    _export_sqlite_table_as_jsonl(
        history_db_path,
        "user_history_profiles",
        run_out_dir / "user_history_profiles_snapshot.jsonl",
    )
    print(f"\nSaved/updated shared run artifacts in: {run_out_dir.resolve()}")
