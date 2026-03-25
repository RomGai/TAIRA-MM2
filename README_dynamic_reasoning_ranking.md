# 模块三：实时偏好推理与精排 (Dynamic Reasoning & Ranking)

实现文件：
- `dynamic_reasoning_ranking_agent.py`（Agent4 + Agent5 管线）
- `reranker.py`（纯 LLM 精排器，Qwen3-8B）

## Agent 4：实时偏好建模专家 (Dynamic Preference Reasoner - LLM)

输入：
- Agent 3 输出中的 `query_relevant_history`（含 `positive`/`negative`）
- 当前 `query`

职责：
- 对比历史正负样本中商品的 `taxonomy / text_tags / visual_tags`。
- 推理当前会话下用户核心诉求，输出结构化偏好条件：
  - `Must_Have`
  - `Nice_to_Have`（若有可分析视觉信息则必须包含视觉偏好结论；若无则不得臆造）
  - `Must_Avoid`
  - `Predicted_Next_Items`（严格 3 个预测，字段含 `item_type` / `likelihood` / `evidence`，且 likelihood 分别为 Most_Likely / Secondary / Possible）
  - `Reasoning`

LLM 调用方式与 Agent3 一致：`Qwen/Qwen3-8B + apply_chat_template(enable_thinking=True)`。

补充：当 `query` 为空时，Agent4 会自动切换到“无 query 专用提示词”，
仅基于相关历史正负行为推理偏好，避免因空 query 引导造成决策噪声。

时序说明：Agent4 会按时间顺序处理 positive 历史；negative 样本通常不具备可靠时序，因此不按其时间先后做序列推理，而作为反偏好证据使用。

## Agent 5：决策精排专家 (Ranking & Scoring Agent - LLM)

输入：
- Agent4 的偏好条件
- Agent3 的候选商品 `candidate_items`

职责：
1. 先执行 `Must_Avoid` 规则过滤（命中即剔除）。
2. 对每个候选商品构造评分 prompt。
3. 使用 Qwen3-8B **下一 token logits** 读取评分 token `1~5` 的概率。
4. 在五档分基础上，叠加与 `Predicted_Next_Items` 的对齐 bonus（Most_Likely > Secondary > Possible）。
5. 最终分数：`ranking_score = logits_weighted_score + prediction_bonus`。
   - `score = Σ(i * P(i))`, i∈{1,2,3,4,5}
6. 依分数降序排序并输出 Top-N。

> 该机制与原 `reranker.py` 中“五档打分 + logits 加权”保持一致，只是改为纯文本 LLM。

可选：传入 `--disable-prediction-bonus` 可关闭预测加分，仅保留 logits 加权分。

## 一体化调用

```python
from dynamic_reasoning_ranking_agent import run_module3
import json

agent3_output = json.load(open("./processed/intent_dual_recall_outputs/xxx.json", "r", encoding="utf-8"))
out = run_module3(agent3_output, model_name="Qwen/Qwen3-8B", top_n=20)
print(out.to_dict())
```

命令行：

```bash
python dynamic_reasoning_ranking_agent.py ./processed/intent_dual_recall_outputs/xxx.json --top-n 20
```
