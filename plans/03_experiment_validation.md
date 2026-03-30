# 子计划 3：实验验证与执行

> 状态：❌ 未开始
>
> 目标：设计高压驱逐场景验证策略差异 + 自动化实验矩阵 + 结果分析可视化

---

## 1. 背景与问题

### 1.1 当前问题

Smoke Test（03/29）结论：20 条短请求在 V100 的 139K token cache 上运行，**0 驱逐发生**，LRU 和 Adaptive 策略表现完全一致。

**原因**：cache 太充裕，从未填满 → 没有驱逐 → 策略差异无法体现。

**必须解决的核心问题**：设计让 cache 填满并频繁驱逐的实验场景。

### 1.2 两种制造 cache 压力的方法

| 方法 | 实现方式 | 优点 | 缺点 |
|------|----------|------|------|
| **限制 cache 大小** | `--num-gpu-blocks-override N` 人为减少可用 block 数 | 简单可控，不需要大量数据 | 不是真实场景 |
| **增加请求量/长度** | 大量长前缀请求自然填满 cache | 更贴近真实 | 需要更多请求和时间 |

**建议**：两种方法结合使用。用 `--num-gpu-blocks-override` 快速验证策略差异，再用真实数据量验证。

---

## 2. 高压驱逐验证

### 2.1 实验设计：受控 cache 压力

**核心思路**：用 `--num-gpu-blocks-override` 将 cache 限制到很小（如 512 blocks = ~8K tokens），然后发送共享前缀的请求流，迫使驱逐发生。

**预期行为差异**：
- **LRU**：盲目驱逐最久未使用的 block，即使是热门 system prompt 的 block
- **Adaptive**：保护高 reuse count 的热门前缀 block，优先驱逐冷 block
- **prefix_match 调度**：优先调度能命中 cache 的请求，减少 cache miss

**实验参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | opt-1.3b | block_size=16 tokens |
| `--num-gpu-blocks-override` | 512 / 1024 / 2048 | 对应 ~8K / ~16K / ~32K tokens |
| 请求数 | 200–500 | 足够产生多轮驱逐 |
| 数据集 | burst_synthetic.jsonl | 70% 热前缀 + 30% 冷 prompt |
| 并发 | 16–32 | 模拟真实并发压力 |

### 2.2 四配置对比

按 proposal 的实验矩阵，每个 cache budget 级别跑 4 种配置：

| # | 配置名 | CLI 参数 |
|---|--------|----------|
| 1 | Baseline | `--eviction-policy lru --scheduling-policy fcfs` |
| 2 | Eviction-only | `--eviction-policy adaptive --scheduling-policy fcfs` |
| 3 | Scheduling-only | `--eviction-policy lru --scheduling-policy prefix_match` |
| 4 | Joint | `--eviction-policy adaptive --scheduling-policy prefix_match` |

### 2.3 关键指标对比

| 指标 | 来源 | 预期差异 |
|------|------|----------|
| **Cache 命中率** | Prometheus `vllm:prefix_cache_hits/queries` | Adaptive > LRU（热前缀被保护） |
| **驱逐数** | Prometheus `vllm:kv_cache_evictions_total` | Adaptive 驱逐更有选择性 |
| **吞吐量** (tokens/s) | benchmark 输出 | Joint > 其他（协同效应） |
| **P99 延迟** (ms) | benchmark 输出 | prefix_match 可能更高（排序开销）或更低（更多 cache 命中） |
| **TTFT** (ms) | benchmark 输出 | 高 cache 命中 → TTFT 降低 |
| **最大等待时间** | Prometheus `vllm:scheduler_max_wait_seconds` | prefix_match 有 aging 保障 |

---

## 3. 自动化实验脚本

### 3.1 `scripts/run_experiment_matrix.py`

自动化运行完整实验矩阵的脚本：

**功能**：
1. 按配置启动 vLLM server（在容器中，覆盖修改后的代码）
2. 等待 server 就绪
3. 用 `vllm bench serve` 或自定义脚本发送请求
4. 收集 Prometheus 指标 + per-request 延迟
5. 关闭 server
6. 重复下一个配置
7. 生成汇总 CSV

**配置矩阵**：

```python
CONFIGS = {
    "baseline":        {"eviction": "lru",      "scheduling": "fcfs"},
    "eviction_only":   {"eviction": "adaptive",  "scheduling": "fcfs"},
    "scheduling_only": {"eviction": "lru",      "scheduling": "prefix_match"},
    "joint":           {"eviction": "adaptive",  "scheduling": "prefix_match"},
}

WORKLOADS = {
    "high_reuse":  {"dataset": "sharegpt", "path": ".../ShareGPT_V3.json", "num_prompts": 200},
    "low_reuse":   {"dataset": "custom",   "path": ".../mmlu_vllm.jsonl",  "num_prompts": 200},
    "burst":       {"dataset": "custom",   "path": ".../burst_synthetic.jsonl", "num_prompts": 500},
}

CACHE_BUDGETS = {
    "small":  512,   # ~8K tokens — 高驱逐压力
    "medium": 2048,  # ~32K tokens — 中等压力
    "large":  8192,  # ~128K tokens — 低压力（接近无限制）
}
```

**总配置数**：4 × 3 × 3 = **36 runs**，每组重复 3 次 → **108 runs**。

### 3.2 `scripts/run_single_experiment.sh`

单次实验的封装脚本（被 matrix 脚本调用）：

```bash
# 输入参数
MODEL=...
EVICTION_POLICY=...
SCHEDULING_POLICY=...
NUM_GPU_BLOCKS=...
DATASET_NAME=...
DATASET_PATH=...
NUM_PROMPTS=...
OUTPUT_DIR=...
PORT=...

# 1. 启动 server
singularity exec --nv --writable-tmpfs $CONTAINER bash -c "
  cp -r $SRC/* $DST/
  cd /tmp
  CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --enable-prefix-caching \
    --eviction-policy $EVICTION_POLICY \
    --scheduling-policy $SCHEDULING_POLICY \
    --num-gpu-blocks-override $NUM_GPU_BLOCKS \
    --max-model-len 2048" &

# 2. 等待 server 就绪
wait_for_server http://localhost:$PORT

# 3. 运行 benchmark
python3 experiments/collect_metrics.py \
  --server http://localhost:$PORT \
  --dataset-name $DATASET_NAME \
  --dataset-path $DATASET_PATH \
  --num-prompts $NUM_PROMPTS \
  --output-dir $OUTPUT_DIR

# 4. 收集 Prometheus 最终快照
curl -s http://localhost:$PORT/metrics > $OUTPUT_DIR/prometheus_final.txt

# 5. 关闭 server
kill %1
```

### 3.3 输出结构

```
experiments/results/
├── matrix_summary.csv              # 所有 runs 的汇总指标
├── 2026-04-XX_baseline_sharegpt_small_r1/
│   ├── requests.jsonl              # per-request 延迟
│   ├── prometheus_final.txt        # Prometheus 指标快照
│   └── summary.json                # 聚合指标
├── 2026-04-XX_joint_burst_medium_r2/
│   └── ...
└── ...
```

---

## 4. 结果分析与可视化

### 4.1 `experiments/analyze_matrix.py`

读取 `matrix_summary.csv`，生成以下图表：

| 图表 | X 轴 | 分组 | Y 轴 | 用途 |
|------|------|------|------|------|
| **策略对比 bar chart** | 4 种配置 | 按工作负载 | 吞吐量 / 命中率 / P99 | 核心结果图 |
| **Cache budget 曲线** | cache 大小 (25/50/75%) | 按配置 | 命中率 / 驱逐数 | 压力敏感性分析 |
| **延迟 CDF** | 延迟 (ms) | 按配置 | 百分位 | P50/P90/P99 对比 |
| **驱逐质量分布** | reuse_count of evicted blocks | 按策略 | 频率 | Adaptive 是否驱逐了低价值 block |
| **时序变化** | 时间 | 按配置 | 命中率 / 驱逐率 | 策略的动态行为 |
| **加速比热力图** | 工作负载 × cache budget | 按配置 | Throughput / Baseline | 论文核心图表 |

### 4.2 预期结果假设

| 场景 | 预期最优配置 | 理由 |
|------|-------------|------|
| 高复用 + 小 cache | Joint | 热前缀被保护（Adaptive）+ 优先调度命中请求（prefix_match） |
| 高复用 + 大 cache | 所有趋同 | cache 充裕，驱逐少，策略差异小 |
| 低复用 + 任意 cache | 所有趋同 | 无前缀共享，所有策略退化为 ~LRU/FCFS |
| 混合/突发 + 小 cache | Joint 或 Eviction-only | 热前缀的驱逐保护最关键 |

---

## 5. 执行步骤

### Phase 1：高压验证（目标 2 天）

- [ ] **Step 1**：编写 `scripts/run_single_experiment.sh` 单次实验封装
- [ ] **Step 2**：手动跑一次高压验证（`--num-gpu-blocks-override 512`，4 配置对比）
  - 确认 Adaptive vs LRU 在高驱逐压力下有可观测差异
  - 确认 prefix_match vs FCFS 在高复用工作负载下有可观测差异
  - 调整参数（cache 大小、请求数、并发数）使差异最大化
- [ ] **Step 3**：记录最优实验参数，确认实验矩阵可行

### Phase 2：自动化实验矩阵（目标 2 天）

- [ ] **Step 4**：编写 `scripts/run_experiment_matrix.py` 自动化矩阵
- [ ] **Step 5**：编写 `experiments/analyze_matrix.py` 分析脚本
- [ ] **Step 6**：小规模试跑（4 配置 × 1 工作负载 × 1 cache budget = 4 runs）
- [ ] **Step 7**：修复问题，确认端到端流程

### Phase 3：完整矩阵运行（目标 3 天，需 GPU）

- [ ] **Step 8**：运行核心矩阵（36 配置 × 3 重复 = 108 runs，~18 GPU 小时）
- [ ] **Step 9**：运行消融实验（~68 runs，~12 GPU 小时）
- [ ] **Step 10**：生成所有图表和汇总数据

---

## 6. GPU 时间预算

| 阶段 | 预估 GPU 小时 | 说明 |
|------|--------------|------|
| 高压验证（手动） | ~2h | 4 配置 × 几种 cache budget |
| 核心矩阵（108 runs） | ~18h | 每 run ~10 分钟 |
| 消融实验（~68 runs） | ~12h | prompt 长度/类型/评分函数 |
| 调试和重跑 | ~10h | 失败重试、参数调整 |
| **总计** | **~42h** | 在 200h 预算内 |

---

## 7. 需要编写的脚本

| 脚本 | 用途 | 优先级 | 状态 |
|------|------|--------|------|
| `scripts/run_single_experiment.sh` | 单次实验：启动 server → benchmark → 收集指标 → 关闭 | P0 | ❌ |
| `scripts/run_experiment_matrix.py` | 自动化跑完整矩阵，生成 CSV 汇总 | P0 | ❌ |
| `experiments/analyze_matrix.py` | 读取汇总 CSV，生成对比图表 | P0 | ❌ |
| `experiments/collect_metrics.py` | 已有，可能需要适配 benchmark 接口 | — | ✅ 已有 |
| `experiments/plot_comparison.py` | 已有，可能需要扩展支持多配置 | — | ✅ 已有 |

---

## 8. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| Adaptive 在高压下仍无显著优势 | 论文结论弱 | 分析原因（如 block_size 太大导致粒度粗）；在 negative result 中讨论 |
| `--num-gpu-blocks-override` 太小导致 OOM | server 崩溃 | 找到最小可用 block 数的下界，每个模型单独标定 |
| 108 runs 部分失败 | 数据不完整 | 自动重试机制；每 run 独立存储，可增量重跑 |
| prefix_match 排序开销在高并发时显著 | P99 延迟恶化 | 单独测量排序开销，在报告中讨论 trade-off |
| Qwen3-8B 在受限 cache 下 OOM | 无法在 V100 上评估 | 用更大 cache budget 或只用 opt-1.3b |

---

## 9. 与其他子计划的关系

```
子计划 0（驱逐策略）──┐
                      ├──→ 子计划 3（实验验证）──→ 子计划 4（报告）
子计划 2（调度器）────┘            ↑
                                   │
子计划 1（数据准备）──────────────┘
```

所有前置工作（策略实现 + 数据集）已完成，本子计划是将它们组合验证的关键阶段。
