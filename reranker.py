"""LLM-based item reranker for Amazon recommendation (Qwen3-8B).

This file implements Agent 5 compatible scoring:
- 5-level relevance buckets (1~5)
- logits-based weighted expectation score

Compared with previous multimodal reranker, this version is pure text LLM,
using product structured profile from module-1 and dynamic constraints from module-3.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def _normalize_prediction_text(v: Any) -> str:
    return " ".join(str(v).strip().lower().split())


class LLMItemReranker:
    """Rerank candidate items with Qwen3-8B via five-level logits weighting."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 8,
        enable_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

        self.id_1 = None
        self.id_2 = None
        self.id_3 = None
        self.id_4 = None
        self.id_5 = None

    def load(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError("transformers/torch are not available for LLMItemReranker.")
        if self._model is not None and self._tokenizer is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        self.id_1 = self._tokenizer.convert_tokens_to_ids("1")
        self.id_2 = self._tokenizer.convert_tokens_to_ids("2")
        self.id_3 = self._tokenizer.convert_tokens_to_ids("3")
        self.id_4 = self._tokenizer.convert_tokens_to_ids("4")
        self.id_5 = self._tokenizer.convert_tokens_to_ids("5")

    @torch.no_grad()
    def _score_with_logits(self, prompt: str) -> Dict[str, Any]:
        self.load()
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        logits = self._model(**inputs).logits[:, -1, :]

        score_logits = torch.stack(
            [
                logits[:, self.id_1],
                logits[:, self.id_2],
                logits[:, self.id_3],
                logits[:, self.id_4],
                logits[:, self.id_5],
            ],
            dim=1,
        )
        probs = torch.nn.functional.softmax(score_logits, dim=1)[0]
        p = probs.tolist()
        weighted = sum((idx + 1) * val for idx, val in enumerate(p))

        return {
            "probs": {"1": p[0], "2": p[1], "3": p[2], "4": p[3], "5": p[4]},
            "weighted_score": float(weighted),
        }

    @staticmethod
    def _build_scoring_prompt(
        query: str,
        preference_constraints: Dict[str, Any],
        item: Dict[str, Any],
    ) -> str:
        must_have = preference_constraints.get("Must_Have", [])
        nice_to_have = preference_constraints.get("Nice_to_Have", [])
        must_avoid = preference_constraints.get("Must_Avoid", [])

        profile = item.get("profile", {})
        compact_item = {
            "item_id": item.get("item_id"),
            "title": profile.get("title", ""),
            "taxonomy": profile.get("taxonomy", {}),
            "text_tags": profile.get("text_tags", {}),
            "visual_tags": profile.get("visual_tags", {}),
            "hypotheses": profile.get("hypotheses", []),
            "overall_confidence": profile.get("overall_confidence", 0.0),
        }

        next_predictions = preference_constraints.get("Predicted_Next_Items", [])
        collaborative_signals = preference_constraints.get("Similar_User_Collaborative_Signals", {})

        return (
            "你是电商推荐精排专家（Agent5）。请从用户视角判断候选商品与当下偏好的匹配程度。\n"
            "评分规则（只能取1~5）：\n"
            "注意：这是 next-item 预测场景，需重点判断该候选是否与 Predicted_Next_Items 一致。\n"
            "1=触碰Must_Avoid或与核心诉求明显冲突；\n"
            "2=弱相关，仅少量满足；\n"
            "3=中等相关，满足部分Must_Have或多个Nice_to_Have；\n"
            "4=高相关，满足大多数Must_Have且有Nice_to_Have加分；\n"
            "5=强匹配，完整满足Must_Have且无冲突，同时在Nice_to_Have表现突出。\n"
            "请只输出一个数字：1/2/3/4/5。\n\n"
            f"用户Query: {query}\n"
            f"Must_Have: {json.dumps(must_have, ensure_ascii=False)}\n"
            f"Nice_to_Have: {json.dumps(nice_to_have, ensure_ascii=False)}\n"
            f"Must_Avoid: {json.dumps(must_avoid, ensure_ascii=False)}\n"
            f"Predicted_Next_Items: {json.dumps(next_predictions, ensure_ascii=False)}\n"
            f"Collaborative_Signals_From_Similar_Users: {json.dumps(collaborative_signals, ensure_ascii=False)}\n"
            f"候选商品画像: {json.dumps(compact_item, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _must_avoid_filter(preference_constraints: Dict[str, Any], item: Dict[str, Any]) -> bool:
        must_avoid = [str(x).strip().lower() for x in preference_constraints.get("Must_Avoid", []) if str(x).strip()]
        if not must_avoid:
            return False

        profile = item.get("profile", {})
        haystacks = [
            profile.get("title", ""),
            json.dumps(profile.get("taxonomy", {}), ensure_ascii=False),
            json.dumps(profile.get("text_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("visual_tags", {}), ensure_ascii=False),
            json.dumps(profile.get("hypotheses", []), ensure_ascii=False),
        ]
        item_text = "\n".join(str(x) for x in haystacks).lower()
        return any(token and token in item_text for token in must_avoid)

    def rerank_items(
        self,
        query: str,
        preference_constraints: Dict[str, Any],
        candidate_items: List[Dict[str, Any]],
        top_n: int = 40,
        disable_prediction_bonus: bool = False,
    ) -> List[Dict[str, Any]]:
        self.load()
        if top_n <= 0:
            return []

        scored: List[Dict[str, Any]] = []
        for item in candidate_items:
            if self._must_avoid_filter(preference_constraints, item):
                continue

            prompt = self._build_scoring_prompt(query, preference_constraints, item)
            score_info = self._score_with_logits(prompt)
            enriched = dict(item)
            enriched["llm_weighted_score"] = score_info["weighted_score"]
            enriched["prediction_bonus"] = 0.0
            enriched["ranking_score"] = float(score_info["weighted_score"])
            enriched["score_probs"] = score_info["probs"]
            scored.append(enriched)

        scored.sort(
            key=lambda x: (
                float(x.get("ranking_score", 0.0)),
                float((x.get("score_probs") or {}).get("5", 0.0)),
            ),
            reverse=True,
        )

        final_items: List[Dict[str, Any]] = []
        for rank, row in enumerate(scored[:top_n], start=1):
            row["rank"] = rank
            final_items.append(row)
        return final_items


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="LLM rerank candidate items from module-3 payload")
    parser.add_argument("input_json", help="JSON containing query/preference_constraints/candidate_items")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--disable-prediction-bonus", action="store_true")
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    reranker = LLMItemReranker(model_name=args.model)
    results = reranker.rerank_items(
        query=str(payload.get("query", "")),
        preference_constraints=dict(payload.get("preference_constraints", {})),
        candidate_items=list(payload.get("candidate_items", [])),
        top_n=args.top_n,
        disable_prediction_bonus=bool(args.disable_prediction_bonus),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
