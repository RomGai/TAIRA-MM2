"""Module 3: Dynamic Reasoning & Ranking for Amazon pipeline.

This module consumes Agent-3 output from `intent_dual_recall_agent.py` and contains:
- Agent 4: Dynamic Preference Reasoner (LLM)
- Agent 5: Ranking & Scoring Agent (LLM)

LLM backbone: Qwen3-8B (text-only), following the same invocation style as Agent 3.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from reranker import LLMItemReranker

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class PreferenceConstraints:
    must_have: List[str]
    nice_to_have: List[str]
    must_avoid: List[str]
    next_item_predictions: List[Dict[str, str]]
    reasoning: str
    collaborative_info: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "Must_Have": self.must_have,
            "Nice_to_Have": self.nice_to_have,
            "Must_Avoid": self.must_avoid,
            "Predicted_Next_Items": self.next_item_predictions,
            "Reasoning": self.reasoning,
        }
        if self.collaborative_info:
            payload.update(self.collaborative_info)
        return payload


@dataclass
class Module3Output:
    user_id: str
    query: str
    preference_constraints: Dict[str, Any]
    ranked_items: List[Dict[str, Any]]
    groundtruth_target_item_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_timestamp(value: Any) -> Tuple[int, int]:
    raw = str(value).strip()
    if not raw:
        return (1, -1)
    try:
        return (0, int(raw))
    except (TypeError, ValueError):
        return (1, -1)


def _sort_history_by_time(history_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(history_rows, key=lambda r: _safe_timestamp(r.get("timestamp")))


def _extract_candidate_item_type_tags(candidate_items: List[Dict[str, Any]], max_tags: int = 120) -> List[str]:
    tags: List[str] = []
    seen = set()
    for item in candidate_items:
        profile = item.get("profile", {}) or {}
        taxonomy = profile.get("taxonomy", {}) or {}
        item_types = taxonomy.get("item_types", [])
        if not isinstance(item_types, list):
            item_types = [item_types]
        for t in item_types:
            tag = str(t).strip()
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
                if len(tags) >= max_tags:
                    return tags
    return tags


def _normalize_preference_phrase(value: Any) -> str:
    s = str(value).strip()
    if not s:
        return ""
    if s.startswith("{") and s.endswith("}") and "item_type" in s:
        import re

        m = re.search(r"item_type[\'\"]?\s*:\s*[\'\"]([^\'\"]+)[\'\"]", s)
        if m:
            return m.group(1).strip()
    s = s.replace("\n", " ").strip("\"'")
    if s.startswith("{") or s.startswith("["):
        return ""
    return " ".join(s.split())


class Qwen3DynamicReasonerLLM:
    """Qwen3 text LLM wrapper for Agent 4 reasoning."""

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
            raise ImportError("transformers/torch are not available for Qwen3DynamicReasonerLLM.")
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

    def infer_constraints(self, query: str, history_rows: List[Dict[str, Any]], candidate_type_tags: List[str]) -> PreferenceConstraints:
        self.load()

        sorted_history_rows = _sort_history_by_time(history_rows)
        history_for_prompt = []
        for row in sorted_history_rows[-120:]:
            history_for_prompt.append(
                {
                    "item_id": row.get("item_id"),
                    "behavior": row.get("behavior"),
                    "timestamp": row.get("timestamp"),
                    "taxonomy": (row.get("profile") or {}).get("taxonomy", {}),
                    "text_tags": (row.get("profile") or {}).get("text_tags", {}),
                    "visual_tags": (row.get("profile") or {}).get("visual_tags", {}),
                    "title": (row.get("profile") or {}).get("title", ""),
                }
            )

        guardrail_block = (
            "先决条件一致性（必须执行）：\n"
            "A) 必须先判断用户当前意图所属的商品类型与关键先决条件，再输出偏好。\n"
            "B) 任何与先决条件冲突的属性必须进入 Must_Avoid，且不得在 Must_Have/Nice_to_Have 中出现冲突项。\n"
            "C) 若用户query存在且历史行为与当前query冲突，以当前query的先决条件优先，历史仅作为风格/预算/题材补充。\n"
            "D) 禁止推荐跨平台或不兼容商品（例如 PC 游戏 vs PS/Xbox/Switch；iOS 配件 vs Android 专用）。\n"
            "E) 对人群敏感类目必须检查目标人群先决条件（例如服饰的性别/年龄段/尺码体系，婴幼儿用品的月龄阶段）。\n"
            "F) 对技术类商品必须检查兼容性先决条件（系统版本、接口/协议、功率/电压、尺寸规格）。\n"
            "G) 若信息不足以确认兼容性，必须在 Must_Avoid 中明确“避免不兼容/平台不符”，并在 Reasoning 写明不确定点。\n"
            "H) 输出前做一次一致性自检：Must_Have 与 Must_Avoid 不能互相矛盾，且所有 Must_Have 都必须满足先决条件。"
        )

        clean_query = (query or "").strip()
        if clean_query:
            prompt = (
                "你是电商推荐系统中的实时偏好建模专家（Agent4）。\n"
                "任务：根据用户当前query与相关历史正负行为，按时间顺序推理用户此刻偏好，并预测用户下一次最可能购买什么。\n"
                "要求：\n"
                "1) 明确区分 Must_Have / Nice_to_Have / Must_Avoid。\n"
                "2) query 中明确提出的硬性要求（如品牌、规格、功效、人群、兼容性、尺寸等）必须优先提取并写入 Must_Have。\n"
                "3) 基于 history（尤其 positive 序列）归纳出的偏好，优先写入 Nice_to_Have。\n"
                "4) 若历史中存在可分析的视觉信息（如 visual_tags/图片衍生描述），Nice_to_Have 必须包含视觉偏好结论；若无可分析视觉信息，则不要引用或臆造不存在的视觉信息。\n"
                "5) history 已按 timestamp 升序给出：positive 需要按时间顺序分析演化偏好与已购买轨迹；negative 样本通常无可靠时序，不要按其时间先后推理。\n"
                "6) 必须结合 history 中 positive 与 negative 的对比证据。\n"
                "7) Must_Have 与 Nice_to_Have 不限制条目数量，由你根据证据充分性自行决定。\n"
                "8) 先决条件必须优先于一般偏好，禁止输出与先决条件冲突的结论。\n"
                "9) 结合候选池 item type tags，输出 Predicted_Next_Items（数组，严格 1 条），且该条 likelihood 必须是 Most_Likely。\n"
                "10) Must_Have/Nice_to_Have/Must_Avoid 的每个元素都必须是简短自然语言偏好短语（如“Nintendo Switch 兼容性”“无线/蓝牙连接”），禁止输出 JSON/字典/键值对/引号包裹的结构化片段。\n"
                f"11) {guardrail_block}\n"
                "12) 输出严格 JSON 对象，字段：Must_Have(数组), Nice_to_Have(数组), Must_Avoid(数组), Predicted_Next_Items(数组), Reasoning(字符串)。\n\n"
                f"当前Query: {clean_query}\n"
                f"候选池ItemTypeTags(JSON): {json.dumps(candidate_type_tags, ensure_ascii=False)}\n"
                f"相关历史记录(JSON): {json.dumps(history_for_prompt, ensure_ascii=False)}"
            )
        else:
            prompt = (
                "你是电商推荐系统中的实时偏好建模专家（Agent4）。\n"
                "任务：当前没有可用query。请仅根据用户相关历史正负行为，按时间顺序推理用户此刻偏好，并预测用户下一次最可能购买什么。\n"
                "要求：\n"
                "1) 不要假设额外query意图，不要引用不存在的query信息。\n"
                "2) 明确区分 Must_Have / Nice_to_Have / Must_Avoid。\n"
                "3) 基于 history（尤其 positive 序列）归纳出的偏好，优先写入 Nice_to_Have。\n"
                "4) 若历史中存在可分析的视觉信息（如 visual_tags/图片衍生描述），Nice_to_Have 必须包含视觉偏好结论；若无可分析视觉信息，则不要引用或臆造不存在的视觉信息。\n"
                "5) history 已按 timestamp 升序给出：positive 需要按时间顺序分析演化偏好与已购买轨迹；negative 样本通常无可靠时序，不要按其时间先后推理。\n"
                "6) 必须结合 history 中 positive 与 negative 的对比证据。\n"
                "7) Must_Have 与 Nice_to_Have 不限制条目数量，由你根据证据充分性自行决定。\n"
                "8) 先决条件必须优先于一般偏好，禁止输出与先决条件冲突的结论。\n"
                "9) 结合候选池 item type tags，输出 Predicted_Next_Items（数组，严格 1 条），且该条 likelihood 必须是 Most_Likely。\n"
                "10) Must_Have/Nice_to_Have/Must_Avoid 的每个元素都必须是简短自然语言偏好短语（如“Nintendo Switch 兼容性”“无线/蓝牙连接”），禁止输出 JSON/字典/键值对/引号包裹的结构化片段。\n"
                f"11) {guardrail_block}\n"
                "12) 输出严格 JSON 对象，字段：Must_Have(数组), Nice_to_Have(数组), Must_Avoid(数组), Predicted_Next_Items(数组), Reasoning(字符串)。\n\n"
                f"候选池ItemTypeTags(JSON): {json.dumps(candidate_type_tags, ensure_ascii=False)}\n"
                f"相关历史记录(JSON): {json.dumps(history_for_prompt, ensure_ascii=False)}"
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

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        payload = self._try_json_decode(content) or {}

        def _normalize_list(key: str) -> List[str]:
            vals = payload.get(key, [])
            if not isinstance(vals, list):
                vals = [vals]
            out = []
            for x in vals:
                p = _normalize_preference_phrase(x)
                if p:
                    out.append(p)
            return out

        raw_preds = payload.get("Predicted_Next_Items", [])
        if not isinstance(raw_preds, list):
            raw_preds = []

        normalized_preds: List[Dict[str, str]] = []
        for row in raw_preds:
            if not isinstance(row, dict):
                continue
            item_type = str(row.get("item_type", "")).strip()
            likelihood = str(row.get("likelihood", "Possible")).strip()
            evidence = str(row.get("evidence", "")).strip()
            if likelihood not in {"Most_Likely", "Secondary", "Possible"}:
                likelihood = "Possible"
            if not item_type:
                continue
            normalized_preds.append(
                {
                    "item_type": item_type,
                    "likelihood": likelihood,
                    "evidence": evidence,
                }
            )

        most_likely = next((row for row in normalized_preds if row["likelihood"] == "Most_Likely"), None)
        if most_likely is None and normalized_preds:
            most_likely = dict(normalized_preds[0])
            most_likely["likelihood"] = "Most_Likely"

        if most_likely is None:
            most_likely = {
                "item_type": candidate_type_tags[0] if candidate_type_tags else "Unknown Item Type",
                "likelihood": "Most_Likely",
                "evidence": "fallback_from_candidate_pool",
            }

        normalized_preds = [most_likely]

        return PreferenceConstraints(
            must_have=_normalize_list("Must_Have"),
            nice_to_have=_normalize_list("Nice_to_Have"),
            must_avoid=_normalize_list("Must_Avoid"),
            next_item_predictions=normalized_preds,
            reasoning=str(payload.get("Reasoning", f"LLM raw output: {content[:800]}")),
        )

    def summarize_collaborative_preferences(
        self,
        query: str,
        current_reasoning: str,
        similar_user_reasonings: List[Dict[str, Any]],
    ) -> str:
        if not similar_user_reasonings:
            return ""
        self.load()
        neighbor_payload = [
            {
                "user_id": str(x.get("user_id", "")),
                "similarity": float(x.get("similarity", 0.0)),
                "reasoning": str(x.get("reasoning", "")),
            }
            for x in similar_user_reasonings
        ]
        prompt = (
            "你是推荐系统协同偏好总结器。请根据当前用户偏好推理和相似用户偏好推理，"
            "总结可迁移的共同偏好，并给出一句可供精排使用的协同信号。\n"
            "输出严格 JSON：{\"shared_preference_summary\":\"...\"}。\n\n"
            f"当前用户query: {query}\n"
            f"当前用户reasoning: {current_reasoning}\n"
            f"相似用户reasonings(JSON): {json.dumps(neighbor_payload, ensure_ascii=False)}"
        )

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=512)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        payload = self._try_json_decode(content) or {}
        summary = str(payload.get("shared_preference_summary", "")).strip()
        if summary:
            return summary
        return f"与当前用户偏好相近的用户通常还会关注：{'; '.join([x.get('reasoning', '')[:60] for x in neighbor_payload[:3]])}"


class CollaborativePreferenceMemory:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preference_memory (
                    user_id TEXT PRIMARY KEY,
                    reasoning TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def upsert(self, user_id: str, reasoning: str, embedding: np.ndarray) -> None:
        emb_text = json.dumps([float(x) for x in embedding.tolist()], ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO preference_memory(user_id, reasoning, embedding, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    reasoning=excluded.reasoning,
                    embedding=excluded.embedding,
                    updated_at=excluded.updated_at
                """,
                (str(user_id), str(reasoning), emb_text, int(time.time())),
            )
            conn.commit()

    def search_similar(self, user_id: str, query_embedding: np.ndarray, similarity_threshold: float, top_k: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT user_id, reasoning, embedding FROM preference_memory WHERE user_id != ?", (str(user_id),))
            for uid, reasoning, emb_text in cur.fetchall():
                try:
                    emb = np.array(json.loads(emb_text), dtype=np.float32)
                except Exception:
                    continue
                sim = self._cosine(query_embedding, emb)
                if sim >= float(similarity_threshold):
                    rows.append({"user_id": str(uid), "reasoning": str(reasoning), "similarity": sim})
        rows.sort(key=lambda x: float(x["similarity"]), reverse=True)
        return rows[: max(1, int(top_k))]


def _reasoning_text_for_embedding(query: str, constraints: PreferenceConstraints) -> str:
    return (
        f"query: {query}\n"
        f"must_have: {json.dumps(constraints.must_have, ensure_ascii=False)}\n"
        f"nice_to_have: {json.dumps(constraints.nice_to_have, ensure_ascii=False)}\n"
        f"must_avoid: {json.dumps(constraints.must_avoid, ensure_ascii=False)}\n"
        f"reasoning: {constraints.reasoning}"
    )


class DynamicPreferenceReasonerAgent:
    """Agent 4: infer structured dynamic constraints from recalled history."""

    def __init__(self, llm: Qwen3DynamicReasonerLLM) -> None:
        self.llm = llm

    def run(self, query: str, query_relevant_history: List[Dict[str, Any]], candidate_type_tags: List[str]) -> PreferenceConstraints:
        return self.llm.infer_constraints(query=query, history_rows=query_relevant_history, candidate_type_tags=candidate_type_tags)


class RankingScoringAgent:
    """Agent 5: rank candidate items with five-level logits weighting."""

    def __init__(self, reranker: LLMItemReranker) -> None:
        self.reranker = reranker

    def run(
        self,
        query: str,
        preference_constraints: PreferenceConstraints,
        candidate_items: List[Dict[str, Any]],
        top_n: int = 40,
        disable_prediction_bonus: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.reranker.rerank_items(
            query=query,
            preference_constraints=preference_constraints.to_dict(),
            candidate_items=candidate_items,
            top_n=top_n,
            disable_prediction_bonus=disable_prediction_bonus,
        )


def run_module3(
    intent_dual_recall_output: Dict[str, Any],
    model_name: str = "Qwen/Qwen3-8B",
    top_n: int = 40,
    disable_must_avoid: bool = False,
    disable_must_have: bool = False,
    disable_prediction_bonus: bool = False,
    save_output: bool = True,
    output_dir: str | Path = "./processed/dynamic_reasoning_ranking_outputs",
    groundtruth_target_item_id: str = "",
    collaborative_db_path: str | Path = "./processed/collaborative_preference_memory.db",
    collaborative_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    collaborative_similarity_threshold: float = 0.5,
    collaborative_top_k: int = 5,
) -> Module3Output:
    """One-shot pipeline from Agent-3 output to module-3 final ranking."""

    query = str(intent_dual_recall_output.get("query", ""))
    user_id = str(intent_dual_recall_output.get("user_id", ""))
    candidate_items = list(intent_dual_recall_output.get("candidate_items", []))
    history_rows = list(intent_dual_recall_output.get("query_relevant_history", []))

    reasoner = DynamicPreferenceReasonerAgent(
        llm=Qwen3DynamicReasonerLLM(model_name=model_name)
    )
    candidate_type_tags = _extract_candidate_item_type_tags(candidate_items)
    constraints = reasoner.run(
        query=query,
        query_relevant_history=history_rows,
        candidate_type_tags=candidate_type_tags,
    )
    collaborative_memory = CollaborativePreferenceMemory(collaborative_db_path)
    collaborative_embedder = SentenceTransformer(collaborative_embedding_model)
    current_reasoning_text = _reasoning_text_for_embedding(query=query, constraints=constraints)
    current_embedding = collaborative_embedder.encode(
        [current_reasoning_text],
        batch_size=1,
        prompt_name="query",
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(np.float32, copy=False)
    similar_users = collaborative_memory.search_similar(
        user_id=user_id,
        query_embedding=current_embedding,
        similarity_threshold=collaborative_similarity_threshold,
        top_k=collaborative_top_k,
    )
    collaborative_summary = reasoner.llm.summarize_collaborative_preferences(
        query=query,
        current_reasoning=constraints.reasoning,
        similar_user_reasonings=similar_users,
    ) if similar_users else ""
    constraints.collaborative_info = {
        "Similar_User_Collaborative_Signals": {
            "similarity_threshold": float(collaborative_similarity_threshold),
            "similar_users": similar_users,
            "shared_preference_summary": collaborative_summary,
        }
    }
    collaborative_memory.upsert(user_id=user_id, reasoning=current_reasoning_text, embedding=current_embedding)

    if disable_must_avoid:
        constraints.must_avoid = []
    if disable_must_have:
        constraints.must_have = []

    ranker = RankingScoringAgent(reranker=LLMItemReranker(model_name=model_name))
    ranked_items = ranker.run(
        query=query,
        preference_constraints=constraints,
        candidate_items=candidate_items,
        top_n=top_n,
        disable_prediction_bonus=disable_prediction_bonus,
    )

    output = Module3Output(
        user_id=user_id,
        query=query,
        preference_constraints=constraints.to_dict(),
        ranked_items=ranked_items,
        groundtruth_target_item_id=str(groundtruth_target_item_id or ""),
    )

    if save_output:
        output_path = Path(output_dir) / f"user_{user_id}_dynamic_reasoning_ranking_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run module-3 dynamic reasoning + ranking")
    parser.add_argument("agent3_output", help="Path to agent-3 output JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--output-dir", default="./processed/dynamic_reasoning_ranking_outputs")
    parser.add_argument("--collaborative-db-path", default="./processed/collaborative_preference_memory.db")
    parser.add_argument("--collaborative-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--collaborative-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--collaborative-top-k", type=int, default=5)
    parser.add_argument("--disable-must-have", action="store_true")
    parser.add_argument("--disable-prediction-bonus", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.agent3_output).read_text(encoding="utf-8"))
    result = run_module3(
        intent_dual_recall_output=payload,
        model_name=args.model,
        top_n=args.top_n,
        output_dir=args.output_dir,
        save_output=True,
        collaborative_db_path=args.collaborative_db_path,
        collaborative_embedding_model=args.collaborative_embedding_model,
        collaborative_similarity_threshold=args.collaborative_similarity_threshold,
        collaborative_top_k=args.collaborative_top_k,
        disable_must_have=bool(args.disable_must_have),
        disable_prediction_bonus=bool(args.disable_prediction_bonus),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
