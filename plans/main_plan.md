# 面向工作负载的 KV Cache 驱逐策略与缓存感知调度 — 详细项目计划

> 最后更新: 2026-03-29（自适应驱逐 + 观测系统 + Smoke Test 全部完成）

---

## 1. 项目概述

基于 vLLM v1 的 prefix caching 基础设施，共同设计**面向工作负载的自适应驱逐策略**和**带公平性控制的缓存感知调度**，评估二者在吞吐量、p99 延迟、缓存命中率和公平性方面的**联合影响**。

与 proposal 对齐的四种实验配置：

| # | 配置 | 驱逐策略 | 调度策略 |
|---|------|----------|----------|
| 1 | Baseline | LRU（vLLM 默认） | FCFS（vLLM 默认） |
| 2 | Eviction-only | Adaptive（ours） | FCFS |
| 3 | Scheduling-only | LRU | Prefix-match（+ aging） |
| 4 | Joint | Adaptive（ours） | Prefix-match（+ aging） |

---

## 2. 当前进度概览（截至 03/29）

### 2.1 已完成

| 组件 | 状态 | 说明 |
|------|------|------|
| **代码库分析** | ✅ | 定位 GPU 端驱逐修改点：`FreeKVCacheBlockQueue`（`kv_cache_utils.py`） |
| **架构理解** | ✅ | 确认 scheduler ↔ 驱逐解耦；prefix cache 查找流程；block 生命周期 |
| **开发分支** | ✅ | `feat/adaptive-eviction-v0.15.1` 基于 `v0.15.1` tag（匹配 Singularity 容器） |
| **项目规划** | ✅ | 本文档；实验矩阵设计；回应 TA 反馈 |
| **GPU 端自适应驱逐策略** | ✅ | 非侵入式设计：`--eviction-policy lru|adaptive`，默认 LRU 零改动。7 文件 +112 行，7 项单元测试通过 |
| **观测系统** | ✅ | 新增驱逐计数器、被驱逐 block reuse_count、prefix cache 利用率；Prometheus 暴露；per-request 收集脚本 + 可视化脚本 |
| **模型准备** | ✅ | `Qwen3-8B`（16GB）+ `opt-1.3b`（7.4GB）已下载至 `opensource/models/` |
| **v0.15.1 版本对齐** | ✅ | 代码回退到 `v0.15.1` 以匹配 Singularity 容器环境；新分支 `feat/adaptive-eviction-v0.15.1` |
| **GPU Smoke Test** | ✅ | opt-1.3b + V100 端到端验证通过。LRU/Adaptive 两种策略均通过 4 项检查（请求完成、cache 查询 > 0、命中率 > 0、重复请求更快）。Prometheus 新增指标（evictions、utilization）正常暴露。结果见 `experiments/results/2026-03-29_smoke_*/` |

### 2.2 未开始

| 组件 | 状态 | 负责 |
|------|------|------|
| **缓存感知调度器** | ❌ | prefix-match 请求重排 + aging 公平性机制 |
| **数据集准备** | ✅ | ShareGPT 92K、MMLU 14K、Burst/Code/RAG 各 500（详见 `plans/01_data_preparation.md`） |
| **高压驱逐验证** | ❌ | 设计大量唯一前缀的 benchmark，使 cache 填满并触发驱逐，验证 Adaptive 与 LRU 的差异 |
| **联合优化实验** | ❌ | 完整实验矩阵运行与分析 |

> **备注**：之前在 `vllm/v1/kv_offload/` 路径上的 LFU/LRU-K/ARC 实现属于 CPU offloading 场景（GPU block 被驱逐后拷贝到 CPU 内存），与本项目目标（GPU 端 prefix cache 驱逐策略优化）不在同一代码路径上，不纳入本项目范围。

> **Smoke Test 结论（03/29）**：两种策略在低压力下表现一致（73.4% 命中率、1.97x 加速、0 驱逐），符合预期——20 条请求远未填满 V100 的 86K token cache。需设计高压 benchmark 触发驱逐才能体现策略差异。

---

## 3. 架构与代码地图

### 3.1 GPU 端 Prefix Cache 驱逐（⬅ 修改目标）

GPU 端 prefix cache 用**硬编码 LRU** 管理 free block 队列。当需要新 block 时，从队列头部取出（最久未使用的），如果该 block 有缓存的 KV 数据，则其 prefix cache 映射被清除——这就是"驱逐"。

```
vllm/v1/core/kv_cache_utils.py
├── KVCacheBlock                     # block 数据结构
│   ├── block_id: int                # 唯一 ID
│   ├── ref_cnt: int                 # 引用计数（>0 表示正在使用）
│   ├── _block_hash: BlockHash       # prefix cache 查找键
│   └── prev/next_free_block         # 双向链表指针
└── FreeKVCacheBlockQueue            # ⬅ 驱逐策略所在地
    ├── popleft()                    # ⬅ 驱逐决策点（当前：总是取链表头 = LRU）
    ├── popleft_n(n)                 # 批量弹出 n 个 block
    ├── append() / append_n()        # 归还 block（放到链表尾 = 标记为最近使用）
    ├── remove()                     # 从链表中间移除（被引用时）
    └── touch()                      # block 被命中时，移到链表尾部

vllm/v1/core/block_pool.py
├── BlockPool
│   ├── get_new_blocks()             # 需要 block 时调用 → free_block_queue.popleft_n()
│   ├── _maybe_evict_cached_block()  # 被弹出的 block 如果有 hash → 清除 prefix cache 映射
│   ├── cache_full_blocks()          # 将完成计算的 block 注册到 prefix cache
│   ├── touch()                      # 命中时：ref_cnt++，从 free queue 移除
│   └── free_blocks()                # 释放时：ref_cnt--，归还到 free queue
└── BlockHashToBlockMap              # hash → block 映射（prefix cache 查找表）
```

**驱逐流程**（当前 LRU）：
```
请求需要 block → BlockPool.get_new_blocks()
  → FreeKVCacheBlockQueue.popleft()        # 取链表头（最久未使用）
    → _maybe_evict_cached_block(block)     # 清除该 block 的 prefix cache 映射
      → block 的 GPU 内存被新数据覆写        # 旧 KV 数据永久丢失
```

**我们的修改目标**：让 `popleft()` 不再盲目取链表头，而是根据 block 的**复用统计**选择最不值得保留的 block 驱逐。

### 3.2 调度器

```
vllm/v1/core/sched/
├── scheduler.py             # 主 Scheduler 类
│   ├── schedule()           # 核心循环：调度 running → 调度 waiting
│   │   ├── kv_cache_manager.get_computed_blocks(req)  # prefix cache 查找
│   │   ├── kv_cache_manager.allocate_slots(req, ...)  # block 分配
│   │   └── _preempt_request(req)                      # OOM 时抢占 running 请求
│   ├── add_request()        # 入队新请求
│   └── finish_requests()    # 完成时释放 KV cache
├── request_queue.py
│   ├── FCFSRequestQueue     # 默认：到达顺序
│   └── PriorityRequestQueue # 按 (priority, arrival_time, request_id) 排序
├── interface.py             # SchedulerInterface 抽象基类
└── output.py                # SchedulerOutput 数据类
```

**关键事实**：Scheduler 与驱逐策略**完全解耦**。Scheduler 调用 `allocate_slots()` → `get_new_blocks()` → `popleft()`，它不知道也不关心驱逐策略是什么。因此：
- 修改驱逐策略 **不需要** 改 scheduler
- 修改调度策略 **不需要** 改驱逐逻辑
- 两者可以**独立开发、独立测试、联合评估**

**缓存感知调度的扩展点**：
1. 在 `request_queue.py` 中新增 `PrefixMatchRequestQueue`，按 prefix match 长度排序
2. 在 `scheduler.py` 的 waiting 请求循环（~第 539 行起）前对请求重排
3. 添加 aging 机制（max-wait 阈值）防止饥饿

### 3.3 现有 Benchmarking 工具

```
vllm/benchmarks/
├── serve.py                           # vllm bench serve — 在线 serving benchmark
├── throughput.py                      # vllm bench throughput — 离线吞吐
├── latency.py                        # vllm bench latency — 单 batch 延迟
└── datasets.py                        # ShareGPT, Random, Sonnet, HF datasets

benchmarks/
├── benchmark_prefix_caching.py        # Prefix caching A/B 测试
├── benchmark_long_document_qa_throughput.py  # 长上下文 prefix 测试
└── multi_turn/
    ├── benchmark_serving_multi_turn.py # 多轮对话 benchmark
    └── convert_sharegpt_to_openai.py   # ShareGPT 格式转换器
```

---

## 4. 自适应驱逐策略设计（⬅ 核心贡献）

### 4.1 设计理念

Proposal 提出："Eviction priority can incorporate **estimated reuse probability**, reuse distance, or workload class, using **online statistics** or lightweight prompt classification to **distinguish high-reuse prefixes from one-off prompts**."

我们选择 **online statistics** 路线：**在线统计每个 block hash 的复用次数**，驱逐时优先移除低复用 block。选择理由：直接观测真实复用模式比间接推断（prompt 分类）更鲁棒，且无需预定义工作负载类别，天然适应未知工作负载分布。

### 4.2 具体方案

**数据结构扩展**：

```python
# 在 KVCacheBlock 或 BlockPool 层面追踪
reuse_counter: dict[BlockHash, int]  # 全局：每个 block hash 被查找的累计次数
```

当 prefix cache 查找命中某个 block hash 时，`reuse_counter[hash] += 1`。

**驱逐决策修改**（替换 `FreeKVCacheBlockQueue.popleft()`）：

```python
def popleft(self) -> KVCacheBlock:
    # 不再总是取链表头（LRU），而是扫描 free list 找 reuse score 最低的 block
    best_victim = None
    best_score = float('inf')
    for block in self._iterate_free_list():
        if block._block_hash is None:
            return block  # 无缓存数据的 block，优先使用（零代价）
        score = self._compute_score(block)
        if score < best_score:
            best_score = score
            best_victim = block
    self._remove_from_list(best_victim)
    return best_victim

def _compute_score(self, block) -> float:
    reuse_count = self.reuse_counter.get(block._block_hash, 0)
    recency = ...  # 基于链表位置或时间戳
    return alpha * reuse_count + (1 - alpha) * recency  # 分数越高越值得保留
```

**自适应特性**：
- **高复用工作负载**（如多轮对话共享 system prompt）：热门前缀 block 的 reuse_count 自然累积，被保护不被驱逐
- **低复用工作负载**（如独立 QA）：所有 block 的 reuse_count 都低，退化为近似 LRU
- **混合工作负载**：自动区分热前缀和冷 prompt，无需显式检测工作负载类型
- **无需 "先探测再切换"**：纯在线统计，持续自适应

### 4.3 性能开销控制

| 关注点 | 应对措施 |
|--------|----------|
| 遍历 free list 开销 | 限制扫描窗口（如只看前 K 个 + 随机采样），或用堆维护 |
| reuse_counter 内存 | BlockHash 数量有限（= 历史出现过的唯一前缀数），内存开销可忽略 |
| 锁 / 并发 | vLLM v1 scheduler 是单线程的，无并发问题 |

### 4.4 评分函数方案（消融实验确定最优）

先实现方案 A 作为 v1，通过消融实验（§6.2）判断是否需要更复杂的方案。

| 方案 | 评分公式 | 特点 |
|------|----------|------|
| **A. 纯复用计数**（v1 默认） | `score = reuse_count` | 最简单，类似 LFU 思想 |
| **B. 加权混合** | `score = α·reuse_count + (1-α)·recency_rank` | 平衡频率和时效性 |
| **C. 衰减复用** | `score = Σ decay^(t - t_i)` for each access t_i | 时间衰减，避免历史高频但已冷却的 block 永驻 |

---

## 5. 回应助教反馈

### 5.1 三类工作负载的数据集选择（Q1）

| 工作负载类型 | 数据集 | 特征 | 前缀共享程度 |
|-------------|--------|------|-------------|
| **高复用** | ShareGPT（多轮对话） | 共享 system prompt，多轮上下文累积 | 高 |
| **高复用** | LMSys-Chat-1M（过滤后） | 真实用户对话，共享常见 system prompt | 高 |
| **低复用** | MMLU / HellaSwag | 独立 QA prompt，极少前缀重叠 | 低 |
| **低复用** | 随机合成数据 | vLLM 内置 `RandomDataset`，无共享 | 无（负对照） |
| **混合/突发** | WildChat | 真实世界混合分布，有时间突发性 | 可变 |
| **混合/突发** | 自定义合成突发数据 | 热前缀 + 冷 prompt 以突发方式到达 | 可控 |

### 5.2 Baselines 与模型选择（Q2）

**Baselines（与 proposal 对齐的 4 配置）：**

| 配置 | 驱逐 | 调度 | 目的 |
|------|------|------|------|
| Baseline | LRU | FCFS | 对照组（vLLM 默认） |
| Eviction-only | Adaptive | FCFS | 量化驱逐策略单独的贡献 |
| Scheduling-only | LRU | Prefix-match | 量化调度策略单独的贡献 |
| Joint | Adaptive | Prefix-match | 量化联合优化的协同效应 |

**模型选择：**

| 模型 | 大小 | 用途 |
|------|------|------|
| `Qwen/Qwen3-8B` | 8B | 主评估模型（本地已有权重） |
| `facebook/opt-1.3b` | 1.3B | 开发调试（V100 上快速迭代） |
| `Qwen/Qwen3-30B-A3B` | 30B MoE | 仅在有 H100 余量时用于大模型验证 |

**硬件（反映 TA 反馈）：**
- **V100 (32GB)**：主开发 + 评估。opt-1.3b 调试，Qwen3-8B 评估。32GB 显存 = 紧凑 cache budget = 高驱逐压力，有利于策略对比。
- **H100 (80GB)**：TA 指出**难以获取**。仅在有余量时用于 Qwen3-30B-A3B 验证，不作为核心实验依赖。
- **GPU 小时预算**：**≤ 200 小时**（TA 建议）。

### 5.3 Prompt 长度与性能提升的关系（Q3）

按 prompt 长度分桶实验，回答："更智能的驱逐/调度策略在何时收益最大？"

| 分桶 | Token 范围 | 假设 |
|------|-----------|------|
| 短 | 64–256 | 缓存压力低 → 各策略趋同 |
| 中 | 256–1024 | 中等压力 → 驱逐策略差异显现 |
| 长 | 1024–4096 | 高压力 → 自适应策略优势明显 |
| 超长 | 4096–16384 | 极端压力 → 仅少数 prompt 能缓存 |

**实现方式**：按长度过滤数据集（ShareGPT、WildChat），每个策略配置在每个分桶中单独运行。

### 5.4 Prompt 类型与性能的关系（Q4）

| Prompt 类型 | 复用模式 | 预期最优策略 | 测试数据 |
|-------------|---------|-------------|----------|
| **纯文本对话** | 高前缀复用（system prompt + 历史） | Adaptive + Prefix-match | ShareGPT 多轮对话 |
| **代码补全** | 极高前缀复用（共享 repo 上下文） | Adaptive + Prefix-match | 合成：共享 repo context + 不同补全 |
| **Agent / Tool calls** | 中等复用（共享工具定义） | Adaptive | 合成 function-calling 格式 |
| **RAG** | 热门文档高复用 | Adaptive | 合成：Zipfian 文档热度 + 不同查询 |
| **独立 QA** | 无复用 | 所有策略 ≈ 相同（负对照） | MMLU 子集 |

---

## 6. 实验矩阵

### 6.1 核心实验（与 proposal Section 4 对齐）

| 维度 | 级别 | 数量 |
|------|------|------|
| 策略配置 | Baseline, Eviction-only, Scheduling-only, Joint | 4 |
| 工作负载 | 高复用, 低复用, 混合/突发 | 3 |
| Cache budget | 25%, 50%, 75% of max GPU blocks | 3 |
| **总配置数** | | **36** |

每组重复 **3 次** → **108 runs**。

V100 上每次 ~10 分钟 → **~18 GPU 小时**（核心矩阵）。

### 6.2 消融实验

| 实验 | 配置数 | 目的 |
|------|--------|------|
| Prompt 长度分桶 | ~24 runs | 各长度分桶下 Adaptive vs Baseline |
| Prompt 类型消融 | ~20 runs | 各 prompt 类型下策略表现 |
| 评分函数对比 | ~18 runs | 方案 A vs B vs C（§4.4） |
| 调度开销测量 | ~6 runs | 测量 prefix-match 排序的额外延迟 |

**总计**: ~108 + ~68 ≈ **176 runs**，约 **30–50 GPU 小时**。

加上调试和开发，**总 GPU 使用量 ~100–150 小时**，在 TA 建议的 **200 小时**内。

### 6.3 报告指标

| 指标 | 说明 | 来源 |
|------|------|------|
| **吞吐量** (tokens/s) | 总输出 token / 总耗时 | benchmark 脚本 |
| **平均延迟** (ms) | 平均 E2E per-request 延迟 | benchmark 脚本 |
| **P99 延迟** (ms) | 第 99 百分位 E2E 延迟 | benchmark 脚本 |
| **TTFT** (ms) | 首 token 时间（均值 + P99） | benchmark 脚本 |
| **Cache 命中率** (%) | Prefix cache hit blocks / total lookup blocks | `PrefixCacheStats` |
| **驱逐速率** (blocks/s) | 每秒驱逐 block 数 | BlockPool 埋点 |
| **最大等待时间** (ms) | 请求在队列中的最长等待 | Scheduler 埋点 |
| **饥饿率** (%) | 超过 max-wait 阈值的请求比例 | Scheduler 埋点 |
| **加速比** | Throughput(策略) / Throughput(Baseline) | 衍生 |

---

## 7. 实施计划

### 阶段 3：自适应驱逐（03/29–04/08）— 当前阶段

#### 3a. 自适应驱逐实现（03/29–04/05）

**需修改的文件：**

| 文件 | 修改内容 |
|------|----------|
| `vllm/config/cache.py` | 新增 `EvictionPolicy` 类型 + `eviction_policy` 字段（默认 `"lru"`） |
| `vllm/engine/arg_utils.py` | 新增 `--eviction-policy` CLI 参数 |
| `vllm/v1/core/kv_cache_utils.py` | `FreeKVCacheBlockQueue` 加 `reuse_counter`、`_popleft_adaptive()`、`increment_reuse()` |
| `vllm/v1/core/block_pool.py` | `touch()` 调用 `increment_reuse()`；驱逐计数器 + cache 利用率 |
| `vllm/v1/core/kv_cache_coordinator.py` | 透传 `eviction_policy`（基类 + 3 子类 + 工厂函数） |
| `vllm/v1/core/kv_cache_manager.py` | 透传 `eviction_policy` + 新增 `prefix_cache_utilization` / `drain_num_evictions` |
| `vllm/v1/core/sched/scheduler.py` | 传递 `eviction_policy` + 填充新 stats 字段 |

**任务清单：**

- [x] 在 `FreeKVCacheBlockQueue` 中增加 `reuse_counter: dict[BlockHashWithGroupId, int]`
- [x] 在 prefix cache 命中路径（`BlockPool.touch()`）中调用 `increment_reuse()` 递增计数
- [x] 修改 `popleft()` → `_popleft_lru()` / `_popleft_adaptive()` 分支
- [x] 添加 `CacheConfig.eviction_policy` + `--eviction-policy` CLI 参数，非侵入式开关
- [x] 配置透传链：`CacheConfig` → `Scheduler` → `KVCacheManager` → `Coordinator` → `BlockPool` → `FreeKVCacheBlockQueue`
- [x] 单元测试：7 项测试通过（LRU baseline、unhashed 优先、lowest reuse eviction、popleft_n、空队列等）
- [x] 用 opt-1.3b 在 V100 上验证功能正确性（Smoke Test 通过，LRU/Adaptive 均正常工作）

#### 3b. 观测系统（03/29–04/05）⬅ 必须做完善

**现有基础设施（可直接复用）：**

| 已有组件 | 位置 | 提供的指标 |
|----------|------|-----------|
| `PrefixCacheStats` | `vllm/v1/metrics/stats.py` | queries/hits 计数（cache 命中率的原始数据） |
| `CachingMetrics` | `vllm/v1/metrics/stats.py` | 滑动窗口 hit_rate 聚合 |
| `KVCacheMetricsCollector` | `vllm/v1/core/kv_cache_metrics.py` | **per-block**: lifetime、idle time、reuse gaps（采样） |
| `PromptTokenStats` | `vllm/v1/metrics/stats.py` | per-iteration: computed vs local_cache_hit 拆分 |
| `FinishedRequestStats` | `vllm/v1/metrics/stats.py` | per-request: `num_cached_tokens` |
| Prometheus 暴露 | `vllm/v1/metrics/loggers.py` | `vllm:prefix_cache_queries/hits`、block histogram |

**需要新增的埋点（已全部完成）：**

| 指标 | 实现位置 | 状态 |
|------|----------|------|
| **驱逐速率** | `BlockPool._num_evictions` + `drain_num_evictions()` → `SchedulerStats.num_evictions` → Prometheus `vllm:kv_cache_evictions_total` | ✅ |
| **Per-request 等待时间** | 已有 `FinishedRequestStats.queued_time`（`scheduled_ts - queued_ts`） | ✅ 复用 |
| **被驱逐 block 的 reuse_count** | `KVCacheEvictionEvent.reuse_count`，在 `_maybe_evict_cached_block()` 中捕获 | ✅ |
| **Cache 利用率** | `BlockPool.get_prefix_cache_utilization()` → `SchedulerStats.prefix_cache_utilization` → Prometheus `vllm:prefix_cache_utilization` | ✅ |
| **Per-request 收集 + 可视化** | `experiments/collect_metrics.py` + `experiments/plot_comparison.py` | ✅ |

**任务清单：**

- [x] 验证 `PrefixCacheStats` 数据链路：`KVCacheManager` → `SchedulerStats` → `StatLogger` → Prometheus（代码追踪确认完整）
- [x] 在 `BlockPool._maybe_evict_cached_block()` 中增加驱逐计数器 + reuse_count 记录
- [x] 在 `KVCacheEvictionEvent` 中增加 `reuse_count` 字段
- [x] 在 `SchedulerStats` 中增加 `num_evictions` + `prefix_cache_utilization`
- [x] Prometheus 新增 `vllm:kv_cache_evictions_total` counter + `vllm:prefix_cache_utilization` gauge
- [x] 文本日志新增 `Prefix cache util: X%` + `Evictions: N`
- [x] 编写 `experiments/collect_metrics.py`：向 vLLM server 发请求，收集 per-request JSONL + Prometheus 汇总
- [x] 编写 `experiments/plot_comparison.py`：LRU vs Adaptive 对比图（bar chart + latency CDF）
- [x] 端到端验证：Smoke Test 确认 Prometheus metrics（102 项指标含新增 evictions/utilization）和日志输出正确

#### 3c. 缓存感知调度器（03/29–04/08）

**需修改的文件：**

| 文件 | 修改内容 |
|------|----------|
| `vllm/v1/core/sched/request_queue.py` | 新增 `PrefixMatchRequestQueue` |
| `vllm/v1/core/sched/scheduler.py` | 在 waiting 请求循环前按 prefix match 长度排序 |

**任务清单：**

- [ ] 实现 `PrefixMatchRequestQueue`（按 prefix cache 命中长度降序）
- [ ] 添加 aging 机制（max-wait 阈值防止饥饿）
- [ ] 接入 `SchedulingPolicy` 枚举，通过配置选择调度策略
- [ ] 单元测试：验证高 prefix 命中请求优先调度 + 长等待请求不被饿死

### 阶段 4：联合评估（04/08–04/19）

| 任务 | 截止日期 |
|------|----------|
| 自动化实验运行脚本 (`scripts/run_experiment_matrix.py`) | 04/08 |
| 准备数据集（ShareGPT, WildChat, MMLU, 合成数据） | 04/09 |
| 运行核心矩阵（36 配置 × 3 重复 = 108 runs） | 04/09–04/14 |
| 运行 prompt 长度消融 | 04/14–04/16 |
| 运行 prompt 类型消融 | 04/14–04/16 |
| 运行评分函数消融 | 04/14–04/16 |
| 数据分析与可视化 | 04/16–04/19 |

### 阶段 5：展示（04/20–04/26）

| 任务 | 截止日期 |
|------|----------|
| 撰写报告 — 引言、相关工作 | 04/22 |
| 撰写报告 — 驱逐策略设计与实现 | 04/22 |
| 撰写报告 — 调度器设计与公平性 | 04/22 |
| 撰写报告 — 评估与结果 | 04/24 |
| 制作幻灯片与 demo | 04/25 |
| 排练 | 04/26 |

---

## 8. 硬件与资源计划

| 资源 | 规格 | 用途 |
|------|------|------|
| **V100 (32GB)** PSC | 主开发与评估 GPU | opt-1.3b 开发，Qwen3-8B 评估 |
| **H100 (80GB)** PSC | 大模型验证（**难获取**） | 仅在有余量时用于 Qwen3-30B-A3B |
| **CPU** | 8–16 vCPUs | 并发客户端模拟 |
| **存储** | ~300GB | 模型权重、实验日志、数据集 |
| **GPU 小时** | **≤ 200h**（TA 建议上限） | 估算 ~100-150h 核心实验 + 调试 |

**V100 上的 cache 压力特性**：32GB 显存对 Qwen3-8B 来说 cache budget 相对紧凑，驱逐频率高，有利于策略对比。这实际上是实验的**优势**而非劣势。

---

## 9. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| `popleft()` 遍历 free list 开销大 | 调度延迟增加 | 限制扫描窗口；用堆维护；单独测量开销 |
| 自适应策略在低复用场景无优势 | 实验结果不显著 | 预期行为（退化为 ~LRU）；用负对照证明不降低性能 |
| Prefix-match 排序计算开销 | 调度循环变慢 | 缓存分数，每 N 步重新评分 |
| WildChat 数据集过大 | Benchmark 启动慢 | 预处理并缓存 tokenized prompts |
| H100 拿不到 | 无法测试大模型 | 核心实验全部在 V100 上完成，30B 实验为可选 |
| GPU 小时不够 | 无法完成完整矩阵 | 优先级：(1) 核心矩阵, (2) 评分函数消融, (3) prompt 消融 |

---

## 10. 已对齐决策

| 决策 | 结论 | 理由 |
|------|------|------|
| 自适应策略方向 | ✅ **Online reuse counting** | 直接观测复用 > 间接推断（prompt 分类）；与 proposal "online statistics" 一致 |
| 调度器是否保留 | ✅ **保留** | Proposal 需要 4 配置实验；调度与驱逐解耦，可独立开发；实现量小 |
| 评分函数 | ✅ **方案 A（纯复用计数）起步** | 通过消融实验（§6.2）判断是否需要 B/C |
| 经典算法（LFU/LRU-K/ARC） | ❌ **不作为我们的策略** | 报告 related work 中讨论；不实现 |

---

## 11. 预期交付物

1. **自适应驱逐策略**：基于在线复用统计的 GPU 端 prefix cache 驱逐
2. **缓存感知调度器**：Prefix-match 调度 + aging 公平性机制
3. **联合评估**：4 配置 × 3 工作负载 × 3 cache budget 完整实验矩阵
4. **Prompt 长度分析**：各长度分桶下的性能提升曲线
5. **Prompt 类型分析**：各工作负载类型下的策略效果
6. **最终报告**：论文质量的 write-up，含图表
7. **展示**：幻灯片 + 策略切换的 live demo
