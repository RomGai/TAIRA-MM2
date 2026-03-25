# Amazon Item Profiler Agents (Qwen3VL-8B)

本说明基于 `process_data.py` 处理后的数据格式，提供两个模块：

- Agent 1: `CandidateItemProfiler`（候选商品解析专家）
- Agent 2: `HistoryItemProfiler`（历史交互商品解析专家）

实现文件：`item_profiler_agents.py`。

## 输入数据与 `process_data.py` 对齐

- `*_item_desc.tsv`: `item_id, image, summary`
- `*_u_i_pairs.tsv`: `user_id, item_id, timestamp`
- `*_user_items_negs.tsv`: `user_id, pos, neg`

## 图片来源说明（你问的重点）

- 两种都支持：
  1. **线上 URL**（例如 `*_item_desc.tsv` 里的 `image` 字段）
  2. **本地文件路径**
- 若传入 URL，`item_profiler_agents.py` 会自动下载并缓存到 `./processed/image_cache/`，后续重复使用同一 URL 会直接命中缓存。

> 最新实现已切换为 Qwen3-VL 官方推荐风格：直接把本地路径或 URL 作为 `messages[].content[].image` 传给 `AutoProcessor.apply_chat_template`。

## 输出数据库

- 全局商品特征库（Global Item DB）
  - sqlite: `global_item_features.db`
  - 表：`global_item_features(item_id, profile_json, updated_at)`

- 用户历史流数据库（User History Log DB）
  - sqlite: `user_history_log.db`
  - 表：`user_history_profiles(user_id, item_id, behavior, timestamp, profile_json, created_at)`

## 关键能力

1. Type-first 细粒度文本标签抽取（`item_type` 必填，`category_path` 可空）
2. 细粒度视觉风格抽取（色彩、版型、风格、材质质感、氛围感等）
3. 输出 JSON 显式保留 `title` 字段并落库
4. Agent 2 增强行为标签与时间戳

## 快速使用

```python
from item_profiler_agents import (
    bootstrap_agents_from_processed,
    ItemProfileInput,
    HistoryItemProfileInput,
)

candidate_profiler, history_profiler = bootstrap_agents_from_processed(
    item_desc_tsv="./processed/Video_Games_item_desc.tsv",
    global_db_path="./processed/global_item_features.db",
    history_db_path="./processed/user_history_log.db",
    model_name="Qwen/Qwen3-VL-8B-Instruct",
)

item = ItemProfileInput(
    item_id="42",
    title="Mechanical Keyboard 75%",
    detail_text="Gasket mount, hot-swappable, RGB, PBT keycaps.",
    main_image="./images/42_main.jpg",
    detail_images=["./images/42_side.jpg", "./images/42_detail.jpg"],
    price="$89.00",
    brand="ABC",
    category_hint="Electronics > Keyboards",
)
candidate_profiler.profile_and_store(item)

hist_item = HistoryItemProfileInput(
    **item.__dict__,
    user_id="1001",
    behavior="positive",
    timestamp=1710000000000,
)
history_profiler.profile_and_store(hist_item)
```

> 注：若环境无 GPU/模型权重，代码可完成模块搭建但无法执行真实推理。

## `__main__` 实际运行行为

直接执行 `python item_profiler_agents.py` 时，会：

1. 从 `*_item_desc.tsv` 按顺序抽取前 5 个商品 `item_id` 跑 Agent 1；
2. 从 `*_user_items_negs.tsv` 读取正负标签，并与 `*_u_i_pairs.tsv` 的时间戳进行关联，选择 2 个用户分别按时间戳升序进行完整序列建模；
3. 将两类 profile 写入本地 SQLite，并在终端打印每条 profile 的 JSON 结果。

- 标签解析时会同时读取 `pos` 和 `neg`；并打印统计信息（总量、成功关联到时间戳的条数、因缺失时间戳被丢弃条数）。
- 历史序列中：positive 严格按 `*_u_i_pairs.tsv` 的原始时间戳升序排序（时间戳小/更早在前）；negative 不要求时间戳，缺失时间戳也会保留并参与建模。
- 对于无时间戳的 negative 样本：profile 中保留 `timestamp=null`，写入 `user_history_profiles` 时使用 `-1` 作为占位时间戳，避免 `int(None)` 异常。

## Qwen3-VL 官方式推理参数

`Qwen3VLExtractor` 读取如下环境变量来控制生成：

- `greedy`（默认 `false`）
- `top_p`（默认 `0.8`）
- `top_k`（默认 `20`）
- `temperature`（默认 `0.7`）
- `repetition_penalty`（默认 `1.0`）
- `presence_penalty`（默认 `1.5`）
- `out_seq_length`（建议映射到 `max_new_tokens`，当前默认 `16384`）

## JSON 解析鲁棒性

- 为降低模型输出非严格 JSON 导致的报错，`Qwen3VLExtractor` 采用多阶段解析：
  1. 直接解析；
  2. 尝试解析 markdown 代码块中的 JSON；
  3. 在原始输出中扫描可解码的 JSON 对象起点。
- 若仍失败，会自动进行严格格式重试（强制只输出单个 JSON 对象、并走更确定性生成）。

## 运行产物落盘（便于人工核验）

`__main__` 统一复用目录 `./processed/profiler_runs/shared/`（不会每次新建目录），并增量更新如下文件：

- `candidate_meta.jsonl`：Agent1 使用的候选商品 meta 输入
- `history_meta.jsonl`：Agent2 使用的历史交互 meta 输入
- `candidate_profiles.jsonl`：Agent1 生成的 profile
- `history_profiles.jsonl`：Agent2 生成的 profile
- `global_item_features_snapshot.jsonl`：全局商品库快照
- `user_history_profiles_snapshot.jsonl`：用户历史库快照

- 若某个 history item 在全局商品库中已经建模（`global_item_features` 已存在该 `item_id`），会直接复用已建模 profile，不重复调用模型。


- 为模拟大批量输入，支持按批次处理与进度打印（环境变量：`batch_size`），即使设置了读取上限条目数也按 batch 语义执行。
- 为避免断点重跑导致重复建模：
  - 候选/历史建模前先查 `global_item_features`，已存在则直接复用；
  - 写入 `user_history_profiles` 前先检查 `(user_id,item_id,behavior,timestamp)`，已存在则跳过插入。
- 关键环境变量补充：
  - `candidate_sample_k`：候选热启动样本数（默认 5）
  - `max_history_rows`：单用户历史最大序列长度（默认 500）
  - `batch_size`：批处理进度粒度（默认 64）
