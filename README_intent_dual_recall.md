# 模块二：意图理解与双路召回 (Intent & Dual Recall)

实现文件：`intent_dual_recall_agent.py`。

本模块直接承接模块一（`item_profiler_agents.py`）的两个数据库输出。

## 1. 先对模块一输出结构做对齐分析

### Global Item DB（全局商品特征库）
- 库表：`global_item_features(item_id, profile_json, updated_at)`
- 关键字段在 `profile_json` 中：
  - `taxonomy.item_type`（商品类型）
  - `taxonomy.category_path`（层级类目路径，数组）
  - 其他文本/视觉标签 (`text_tags`, `visual_tags`) 作为召回后可用特征

### User History Log DB（用户历史流数据库）
- 库表：`user_history_profiles(user_id, item_id, behavior, timestamp, profile_json, created_at)`
- 关键字段：
  - `behavior`（positive/negative）
  - `timestamp`（毫秒时间戳）
  - `profile_json` 中同样包含 `taxonomy.item_type`、`taxonomy.category_path`

> 因此模块二的“相关性主键”采用：`taxonomy.category_path` + `taxonomy.item_type`。

## 2. Agent 3（Routing & Recall Agent - LLM）设计

输入：
- `user_id`
- 用户实时 `query`

> 若 `query` 为空，Agent 3 会自动切换到“基于用户历史自主意图推断”模式。

### 职责 1：意图与层级映射
1. 从全局商品库扫描得到：
   - 全部 `category_path` 清单
   - 全部 `item_type` 清单
2. 把清单喂给 Qwen3（非 VL），让模型输出：
   - `category_paths`（二维数组）
   - `item_types`（数组）
   - `reasoning`
3. 若都不匹配，允许模型返回新造类目路径。

### 无 Query 时的兜底策略（历史自驱动）
当 `query` 为空时，不调用 LLM 路由，改为：
1. 读取该用户最近 `lookback` 条历史（默认 200）。
2. 优先使用 `positive` 行为记录；若没有，再回退到全部近期记录。
3. 按 `taxonomy.category_path` 与 `taxonomy.item_type` 做频次统计，取 top 类目/类型作为本次路由目标。
4. 路 B 在该模式下不再做相关性过滤，而是返回该用户近期**全部历史记录**（按时间倒序）。

### 职责 2：动态上卷双路召回

#### 路 A：全局商品召回
- 先按 LLM 选择的类目路径/类型精确召回。
- 若召回量 `< min_candidate_items`，对每条层级路径做“去掉最后一级”的上卷。
- 重复上卷直至数量满足或无法继续。


#### 路 A 去重策略（新增）
- 默认开启 `exclude_seen_items=True`：
  - 会基于用户在 `user_history_profiles` 中的**原始全历史序列**（不依赖路 B 的相关性过滤）构建 `seen_item_ids`；
  - 从路 A 候选集中剔除这些已交互 `item_id`，避免把“看过/点过/负反馈过”的旧商品再次推给用户。
- 可通过参数控制：
  - `exclude_seen_items`：是否开启历史去重（默认 `True`）
  - `seen_history_lookback`：构建 `seen_item_ids` 时最多读取多少历史条目（默认 `5000`）

#### 路 B：用户历史精准召回
- 在 `user_history_profiles` 中按 `user_id` 拉取最近记录。
- 使用与路 A 相同的类目/类型匹配逻辑过滤。
- 返回 query 高相关历史交互记录（保留 `behavior` 和 `timestamp`）。

## 3. Qwen3 调用方式

代码遵循你提供的官方范式：
- `AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")`
- `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto", device_map="auto")`
- `apply_chat_template(..., enable_thinking=True)`
- `generate(max_new_tokens=...)`
- 解析 `</think>` 对应 token id `151668`

## 4. 模块输出

`RoutingRecallAgent.run(...)` 输出结构：
- `candidate_items`: 路 A 召回商品集合
- `query_relevant_history`: 路 B 历史记录（有 query 时为相关子集；无 query 时为全部历史）
- `routing`: 本次路由说明（选中类目、item_type、最终上卷层级）

其中 `routing.reasoning` 会明确说明本次是 `LLM 路由` 还是 `query 为空时的历史推断`。

可直接作为下一阶段排序/重排输入。
补充参数：
- `interested_item_types_k`（默认 `3`）：无论 query 是否为空，都会结合用户历史推断 top-k 兴趣 `item_type`，并与路由类型合并后用于路 A 商品召回。


## 5. 最小使用示例

```python
from intent_dual_recall_agent import Qwen3RouterLLM, GlobalHistoryAccessor, RoutingRecallAgent

llm = Qwen3RouterLLM(model_name="Qwen/Qwen3-8B")
accessor = GlobalHistoryAccessor(
    global_db_path="./processed/global_item_features.db",
    history_db_path="./processed/user_history_log.db",
)
agent = RoutingRecallAgent(llm=llm, accessor=accessor)

result = agent.run(
    user_id="123",
    query="想买适合两个人客厅联机的体感游戏",
    min_candidate_items=20,
)

print(result.candidate_items[:3])
print(result.query_relevant_history[:3])
```

## 6. 与 `test/microlens/test_with_llava.py` 的评测口径对齐建议

`test_with_llava.py` 在计算排序指标前，会将预测分数 reshape 为 `(-1, 21)`，即默认每个评测单元固定是 21 个候选样本（通常是 1 个正样本 + 20 个负样本）。

因此若你希望 `run_full_agents_pipeline.py` 的产出与该评测方式做**条件一致**对齐，推荐按以下口径：

1. **用户历史建模保持完整**：
   - Agent 2 仍然读取用户历史并写入 `user_history_profiles`；
   - Agent 4 的偏好推理继续基于完整历史（或 query 相关历史）进行，不要把历史截断成 21 条。
2. **候选评测池限制为 21 个样本**：
   - 在评测阶段，为每个用户/样本单元提供固定 21 个待排候选；
   - Agent 3/5 的召回与重排应当只在这 21 个候选里进行，以保证与 `reshape(-1, 21)` 的指标定义一致。
3. **指标按组计算**：
   - 最终按每组 21 个候选计算 Recall@K / MRR@K / NDCG@K，避免与“全库召回”场景混算。

简言之：
> 对齐 `test_with_llava.py` 时，应该“**候选库受限（21）** + **历史建模照常** + **在受限候选上做召回/重排并计算分组指标**”。

## 7. 直接可用脚本：先构造 Agent1 的 21 样本全量商品库，再跑全流程

新增脚本：`run_full_agents_pipeline_eval21.py`

特点：
- 在 Agent 1 之前，先构造 1 正 + 20 负的 21-item 商品库（并把这 21 个 item 当作 Agent 1 的全量商品输入）。
- Agent 2 仍使用原始全量历史输入文件，不做 21 截断。
- 构造出的 21-item 库会输出为 `--prepared-item-desc-out`，并记录元信息到 `--eval-unit-meta-out`。

示例：

```bash
python run_full_agents_pipeline_eval21.py \
  --item-desc-tsv ./processed/Video_Games_item_desc.tsv \
  --user-pairs-tsv ./processed/Video_Games_u_i_pairs.tsv \
  --eval-user-items-negs-tsv ./processed/Video_Games_user_items_negs_test.csv \
  --agent2-user-items-negs-tsv ./processed/Video_Games_user_items_negs.tsv \
  --target-user-id 23 \
  --positive-selection latest \
  --exclude-seen-for-negatives \
  --bundle-output ./processed/full_pipeline_eval21_bundle.zip
```

说明：
- `--eval-user-items-negs-tsv` 用于挑选“评测单元”（用户 + 正样本 + 负样本来源）。
- 若该文件负样本不足 20，脚本会从全商品池中按随机种子补齐到 20（可选排除该用户历史看过商品）。
- 最终确保 Agent 1 的商品输入恰好 21 个 item。
- 正样本默认使用 `--positive-selection latest`：从该用户正样本集合中按 `Video_Games_u_i_pairs.tsv` 的时间戳选择“最晚一次交互”的 item。
- 若想回退旧行为，可改用 `--positive-selection index --positive-index N`。
- 若你希望 next-item 预测更开放（避免硬约束过强），可加 `--disable-must-have`，使 Agent5 不使用 Must_Have 作为强约束。
- 若你希望暂时关闭预测加分影响，可加 `--disable-prediction-bonus`，使 Agent5 仅按 logits 加权分排序。
- 脚本按用户循环处理；每处理完一个 user 会实时打印当前累计指标（AUC / Recall@K / MRR@K / NDCG@K），指标公式与 `test_with_llava.py` 保持一致的分组口径。


补充（共享商品库复用）：
- 可通过 `--shared-global-db-path` 指定一个共享的商品库 DB（`global_item_features.db`），所有 user 复用同一库。
- 可通过 `--shared-history-db-path` 指定共享历史库，避免重复写入历史画像。
- 可通过 `--agent2-item-desc-tsv` 为 Agent2 提供完整商品元数据回退源；当当前 21-item 输入中不存在某个历史 item 时，Agent2 会从该文件取 `image/summary`，并在全局商品库缺失时按 Agent1 逻辑补建该 item 的画像。
