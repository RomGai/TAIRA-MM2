"""Intent understanding and dual-recall module (Agent 3) for Amazon pipeline.

This module directly connects to outputs of `item_profiler_agents.py`:
- Global item DB: `global_item_features(item_id, profile_json, updated_at)`
- User history DB: `user_history_profiles(user_id, item_id, behavior, timestamp, profile_json, created_at)`

Agent 3 (Routing & Recall Agent - LLM) responsibilities:
1) Parse real-time query and map it to categories/item types.
2) Route A: recall candidate items from global DB with dynamic hierarchical roll-up.
3) Route B: recall query-relevant user history records from history DB.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class RoutingResult:
    query: str
    category_paths: List[List[str]]
    item_types: List[str]
    reasoning: str


@dataclass
class IntentDualRecallOutput:
    query: str
    user_id: str
    routing: Dict[str, Any]
    candidate_items: List[Dict[str, Any]]
    query_relevant_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sanitize_for_filename(value: str) -> str:
    safe = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)]
    return "".join(safe).strip("_") or "unknown"


def _build_output_file_path(
    user_id: str,
    query: str,
    output_dir: str | Path = "./processed/intent_dual_recall_outputs",
) -> Path:
    query_tag = _sanitize_for_filename((query or "no_query")[:40])
    filename = f"user_{_sanitize_for_filename(user_id)}_{query_tag}_intent_dual_recall_output.json"
    return Path(output_dir) / filename


class Qwen3RouterLLM:
    """Qwen3 (text-only) wrapper following official usage style."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 2048,
        enable_thinking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError("transformers/torch are not available for Qwen3RouterLLM.")
        if self._model is not None and self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    @staticmethod
    def _try_json_decode(text: str) -> Optional[Dict[str, Any]]:
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        if "```" in stripped:
            for part in stripped.split("```"):
                cand = part.replace("json", "", 1).strip()
                if not cand:
                    continue
                try:
                    payload = json.loads(cand)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    continue
        return None

    def route(
        self,
        query: str,
        category_catalog: Sequence[str],
        item_type_catalog: Sequence[str],
    ) -> RoutingResult:
        self.load()

        catalog_text = "\n".join(f"- {c}" for c in category_catalog[:300])
        item_type_text = "\n".join(f"- {i}" for i in item_type_catalog[:300])
        prompt = (
            "你是电商检索路由专家。请把用户Query映射到给定类目/类型；若都不匹配，可新造一个合理类目。\n"
            "输出必须是一个JSON对象，字段:"
            "category_paths(二维数组，每条是层级路径), item_types(数组), reasoning(字符串)。\n\n"
            f"用户Query: {query}\n\n"
            "候选类目路径清单:\n"
            f"{catalog_text}\n\n"
            "候选item_type清单:\n"
            f"{item_type_text}\n"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Parse thinking content boundary token (official pattern).
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        payload = self._try_json_decode(content)
        if payload is None:
            payload = {
                "category_paths": [],
                "item_types": [],
                "reasoning": f"Failed to parse JSON from LLM output: {content[:500]}",
            }

        raw_paths = payload.get("category_paths", [])
        category_paths: List[List[str]] = []
        for p in raw_paths:
            if isinstance(p, list):
                segs = [str(x).strip() for x in p if str(x).strip()]
            else:
                segs = [x.strip() for x in str(p).replace("/", ">").split(">") if x.strip()]
            if segs:
                category_paths.append(segs)

        item_types = [str(x).strip() for x in payload.get("item_types", []) if str(x).strip()]
        return RoutingResult(
            query=query,
            category_paths=category_paths,
            item_types=item_types,
            reasoning=str(payload.get("reasoning", "")),
        )


class GlobalHistoryAccessor:
    """Read-only accessor over module-1 output databases."""

    def __init__(self, global_db_path: str | Path, history_db_path: str | Path) -> None:
        self.global_conn = sqlite3.connect(str(global_db_path))
        self.history_conn = sqlite3.connect(str(history_db_path))
        self.global_conn.row_factory = sqlite3.Row
        self.history_conn.row_factory = sqlite3.Row

    @staticmethod
    def _extract_taxonomy(profile: Dict[str, Any]) -> Tuple[List[str], str]:
        taxonomy = profile.get("taxonomy", {}) if isinstance(profile, dict) else {}
        path = taxonomy.get("category_path", [])
        if not isinstance(path, list):
            path = []
        clean_path = [str(x).strip() for x in path if str(x).strip()]
        item_type = str(taxonomy.get("item_type", "")).strip()
        return clean_path, item_type

    def category_catalog(self) -> Tuple[List[str], List[str]]:
        categories: set[str] = set()
        item_types: set[str] = set()
        rows = self.global_conn.execute("SELECT profile_json FROM global_item_features").fetchall()
        for row in rows:
            profile = json.loads(row["profile_json"])
            path, item_type = self._extract_taxonomy(profile)
            if path:
                categories.add(" > ".join(path))
            if item_type:
                item_types.add(item_type)
        return sorted(categories), sorted(item_types)

    @staticmethod
    def _is_relevant(
        profile: Dict[str, Any],
        target_paths: Sequence[Sequence[str]],
        target_item_types: Sequence[str],
    ) -> bool:
        path, item_type = GlobalHistoryAccessor._extract_taxonomy(profile)
        path_lower = [x.lower() for x in path]
        type_lower = item_type.lower()

        for tp in target_paths:
            tp_lower = [x.lower() for x in tp]
            if tp_lower and len(path_lower) >= len(tp_lower) and path_lower[: len(tp_lower)] == tp_lower:
                return True
            if tp_lower and tp_lower == path_lower:
                return True

        for t in target_item_types:
            if t.lower() and t.lower() == type_lower:
                return True
        return False

    def recall_global_items(
        self,
        target_paths: Sequence[Sequence[str]],
        target_item_types: Sequence[str],
        min_items: int = 20,
        max_items: int = 200,
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        rows = self.global_conn.execute(
            "SELECT item_id, profile_json, updated_at FROM global_item_features"
        ).fetchall()

        all_items = []
        for row in rows:
            profile = json.loads(row["profile_json"])
            all_items.append(
                {
                    "item_id": row["item_id"],
                    "profile": profile,
                    "updated_at": row["updated_at"],
                }
            )

        rollup_paths = [list(p) for p in target_paths]
        recalled: List[Dict[str, Any]] = []
        seen_item_ids: set[str] = set()

        def add_matches(paths: Sequence[Sequence[str]]) -> None:
            nonlocal recalled
            for item in all_items:
                if item["item_id"] in seen_item_ids:
                    continue
                if self._is_relevant(item["profile"], paths, target_item_types):
                    recalled.append(item)
                    seen_item_ids.add(item["item_id"])
                    if len(recalled) >= max_items:
                        return

        add_matches(rollup_paths)
        while len(recalled) < min_items:
            rolled = []
            for p in rollup_paths:
                if len(p) > 1:
                    rolled.append(p[:-1])
            rolled = [p for i, p in enumerate(rolled) if p and p not in rolled[:i]]
            if not rolled:
                break
            rollup_paths = rolled
            add_matches(rollup_paths)
            if len(recalled) >= max_items:
                break

        return recalled[:max_items], rollup_paths

    def fetch_global_items_by_ids(
        self,
        item_ids: Sequence[str],
        max_items: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch global catalog items by provided item-id order."""
        if not item_ids:
            return []

        rows = self.global_conn.execute(
            "SELECT item_id, profile_json, updated_at FROM global_item_features"
        ).fetchall()
        row_map: Dict[str, Any] = {str(r["item_id"]): r for r in rows}

        out: List[Dict[str, Any]] = []
        for item_id in item_ids:
            row = row_map.get(str(item_id))
            if row is None:
                continue
            out.append(
                {
                    "item_id": str(row["item_id"]),
                    "profile": json.loads(row["profile_json"]),
                    "updated_at": row["updated_at"],
                }
            )
            if len(out) >= max_items:
                break
        return out

    def recall_user_history(
        self,
        user_id: str,
        target_paths: Sequence[Sequence[str]],
        target_item_types: Sequence[str],
        max_rows: int = 300,
    ) -> List[Dict[str, Any]]:
        rows = self.history_conn.execute(
            """
            SELECT user_id, item_id, behavior, timestamp, profile_json, created_at
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(max_rows) * 3),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            profile = json.loads(row["profile_json"])
            if not self._is_relevant(profile, target_paths, target_item_types):
                continue
            results.append(
                {
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "behavior": row["behavior"],
                    "timestamp": row["timestamp"],
                    "profile": profile,
                    "created_at": row["created_at"],
                }
            )
            if len(results) >= max_rows:
                break
        return results

    def recall_user_history_all(
        self,
        user_id: str,
        max_rows: int = 300,
    ) -> List[Dict[str, Any]]:
        """Return all recent history rows for the user (no relevance filtering)."""
        rows = self.history_conn.execute(
            """
            SELECT user_id, item_id, behavior, timestamp, profile_json, created_at
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(max_rows)),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            profile = json.loads(row["profile_json"])
            results.append(
                {
                    "user_id": row["user_id"],
                    "item_id": row["item_id"],
                    "behavior": row["behavior"],
                    "timestamp": row["timestamp"],
                    "profile": profile,
                    "created_at": row["created_at"],
                }
            )
        return results

    def user_seen_item_ids(
        self,
        user_id: str,
        lookback: int = 5000,
    ) -> set[str]:
        """Return deduplicated item_ids from the user's full raw history sequence."""
        rows = self.history_conn.execute(
            """
            SELECT item_id
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(lookback)),
        ).fetchall()
        return {str(r["item_id"]) for r in rows if str(r["item_id"]).strip()}

    def _top_item_types_from_history(
        self,
        user_id: str,
        top_k: int = 3,
        lookback: int = 300,
    ) -> List[str]:
        """Infer top-k interested item types from recent history."""
        rows = self.history_conn.execute(
            """
            SELECT profile_json
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(lookback)),
        ).fetchall()
        type_cnt: Dict[str, int] = {}
        for row in rows:
            try:
                profile = json.loads(row["profile_json"])
            except json.JSONDecodeError:
                continue
            _path, item_type = self._extract_taxonomy(profile)
            if item_type:
                type_cnt[item_type] = type_cnt.get(item_type, 0) + 1
        ranked = sorted(type_cnt.items(), key=lambda x: (-x[1], x[0]))
        return [t for t, _ in ranked[: max(1, int(top_k))]]

    def infer_user_intent_from_history(
        self,
        user_id: str,
        lookback: int = 200,
        min_positive_first: bool = True,
        top_category_paths_k: int = 3,
        top_item_types_k: int = 3,
    ) -> RoutingResult:
        """Infer category intent from user history when query is empty.

        Strategy:
        1) Prefer recent positive interactions (if available)
        2) Fallback to all recent interactions
        3) Aggregate frequent taxonomy.category_path / taxonomy.item_type
        """
        rows = self.history_conn.execute(
            """
            SELECT behavior, timestamp, profile_json
            FROM user_history_profiles
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (str(user_id), int(lookback)),
        ).fetchall()

        if not rows:
            return RoutingResult(
                query="",
                category_paths=[],
                item_types=[],
                reasoning="No query and no history found; cannot infer user intent.",
            )

        scoped_rows = rows
        if min_positive_first:
            positives = [r for r in rows if str(r["behavior"]).lower() == "positive"]
            if positives:
                scoped_rows = positives

        cat_cnt: Dict[str, int] = {}
        type_cnt: Dict[str, int] = {}
        for r in scoped_rows:
            try:
                profile = json.loads(r["profile_json"])
            except json.JSONDecodeError:
                continue
            path, item_type = self._extract_taxonomy(profile)
            if path:
                key = " > ".join(path)
                cat_cnt[key] = cat_cnt.get(key, 0) + 1
            if item_type:
                type_cnt[item_type] = type_cnt.get(item_type, 0) + 1

        top_cats = sorted(cat_cnt.items(), key=lambda x: (-x[1], x[0]))[
            : max(1, int(top_category_paths_k))
        ]
        top_types = sorted(type_cnt.items(), key=lambda x: (-x[1], x[0]))[: max(1, int(top_item_types_k))]
        paths = [[seg.strip() for seg in cat.split(">") if seg.strip()] for cat, _ in top_cats]
        item_types = [t for t, _ in top_types]

        reason_scope = "positive-only" if scoped_rows is not rows else "all-recent"
        return RoutingResult(
            query="",
            category_paths=paths,
            item_types=item_types,
            reasoning=(
                "Query is empty; inferred intent from "
                f"user history ({reason_scope}, samples={len(scoped_rows)})."
            ),
        )


class RoutingRecallAgent:
    """Agent 3: routing + dual recall."""

    def __init__(self, llm: Qwen3RouterLLM, accessor: GlobalHistoryAccessor) -> None:
        self.llm = llm
        self.accessor = accessor

    def run(
        self,
        user_id: str,
        query: str,
        min_candidate_items: int = 20,
        max_candidate_items: int = 200,
        max_history_rows: int = 200,
        history_category_paths_k: int = 3,
        query_category_paths_k: int = 3,
        interested_item_types_k: int = 3,
        exclude_seen_items: bool = True,
        seen_history_lookback: int = 5000,
        filter_candidates_by_item_type: bool = True,
        candidate_item_ids_scope: Optional[Sequence[str]] = None,
        save_output: bool = True,
        output_dir: str | Path = "./processed/intent_dual_recall_outputs",
    ) -> IntentDualRecallOutput:
        clean_query = (query or "").strip()
        category_catalog, item_type_catalog = self.accessor.category_catalog()
        if clean_query:
            routing = self.llm.route(clean_query, category_catalog, item_type_catalog)
        else:
            routing = self.accessor.infer_user_intent_from_history(
                user_id=user_id,
                top_category_paths_k=history_category_paths_k,
                top_item_types_k=interested_item_types_k,
            )

        if clean_query:
            routing.category_paths = routing.category_paths[: max(1, int(query_category_paths_k))]

        history_top_item_types = self.accessor._top_item_types_from_history(
            user_id=user_id,
            top_k=interested_item_types_k,
        )

        merged_item_types: List[str] = []
        for t in [*routing.item_types, *history_top_item_types]:
            if t and t not in merged_item_types:
                merged_item_types.append(t)
        routing.item_types = merged_item_types[: max(1, int(interested_item_types_k))]

        if not routing.category_paths and routing.item_types:
            routing.category_paths = [[routing.item_types[0]]]

        if filter_candidates_by_item_type:
            candidate_items, final_rollup_paths = self.accessor.recall_global_items(
                routing.category_paths,
                routing.item_types,
                min_items=min_candidate_items,
                max_items=max_candidate_items,
            )
        else:
            candidate_items = self.accessor.fetch_global_items_by_ids(
                item_ids=list(candidate_item_ids_scope or []),
                max_items=max_candidate_items,
            )
            final_rollup_paths = [list(p) for p in routing.category_paths]

        if clean_query:
            history_rows = self.accessor.recall_user_history(
                user_id=user_id,
                target_paths=final_rollup_paths,
                target_item_types=routing.item_types,
                max_rows=max_history_rows,
            )
        else:
            history_rows = self.accessor.recall_user_history_all(
                user_id=user_id,
                max_rows=max_history_rows,
            )

        seen_item_ids: set[str] = set()
        if exclude_seen_items and filter_candidates_by_item_type:
            seen_item_ids = self.accessor.user_seen_item_ids(
                user_id=user_id,
                lookback=seen_history_lookback,
            )
            if seen_item_ids:
                candidate_items = [
                    x for x in candidate_items if str(x.get("item_id", "")) not in seen_item_ids
                ]

        output = IntentDualRecallOutput(
            query=clean_query,
            user_id=str(user_id),
            routing={
                "reasoning": routing.reasoning,
                "selected_category_paths": routing.category_paths,
                "selected_item_types": routing.item_types,
                "history_category_paths_k": max(1, int(history_category_paths_k)),
                "query_category_paths_k": max(1, int(query_category_paths_k)),
                "interested_item_types_k": max(1, int(interested_item_types_k)),
                "history_top_item_types": history_top_item_types,
                "exclude_seen_items": bool(exclude_seen_items),
                "seen_history_lookback": int(seen_history_lookback),
                "seen_item_count": len(seen_item_ids),
                "final_rollup_paths": final_rollup_paths,
                "filter_candidates_by_item_type": bool(filter_candidates_by_item_type),
                "candidate_item_scope_size": len(candidate_item_ids_scope or []),
                "catalog_size": {
                    "category_paths": len(category_catalog),
                    "item_types": len(item_type_catalog),
                },
            },
            candidate_items=candidate_items,
            query_relevant_history=history_rows,
        )

        if save_output:
            path = _build_output_file_path(str(user_id), clean_query, output_dir=output_dir)
            output.routing["saved_output_path"] = str(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(output.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        return output


if __name__ == "__main__":
    # Example usage:
    # python intent_dual_recall_agent.py
    # Requires module-1 DBs generated beforehand.
    llm = Qwen3RouterLLM(model_name="Qwen/Qwen3-8B")
    accessor = GlobalHistoryAccessor(
        global_db_path="./processed/global_item_features.db",
        history_db_path="./processed/user_history_log.db",
    )
    agent = RoutingRecallAgent(llm=llm, accessor=accessor)

    out = agent.run(
        user_id="0",
        query="", #我想找适合客厅多人玩的体感游戏
        min_candidate_items=10,
        query_category_paths_k=2,
        history_category_paths_k=3,
        save_output=True,
        output_dir="./processed/intent_dual_recall_outputs",
    )
    print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2, default=str))
