# 子计划 0：自适应驱逐策略 + 观测系统

> 状态：✅ 全部完成（03/29）
>
> 目标：在 vLLM v1 GPU 端 prefix cache 中实现基于在线复用统计的自适应驱逐策略，并搭建完整的观测系统

---

## 1. 背景

vLLM v1 的 prefix cache 使用**硬编码 LRU** 管理 free block 队列。当需要新 block 时，从队列头部取出最久未使用的 block，清除其 prefix cache 映射——即"驱逐"。该策略无法区分高复用前缀（如 system prompt）和一次性 prompt，导致热前缀被误驱逐。

**核心修改目标**：让 `FreeKVCacheBlockQueue.popleft()` 不再盲目取链表头，而是根据 block 的**在线复用统计**选择最不值得保留的 block 驱逐。

---

## 2. 自适应驱逐策略

### 2.1 设计方案

- **数据结构**：在 `FreeKVCacheBlockQueue` 中新增 `reuse_counter: dict[BlockHashWithGroupId, int]`
- **计数更新**：prefix cache 命中时（`BlockPool.touch()`）调用 `increment_reuse()` 递增
- **驱逐决策**：扫描 free list，优先使用无缓存数据的 block（零代价），其次驱逐 reuse_count 最低的 block
- **非侵入式开关**：`--eviction-policy lru|adaptive`，默认 LRU 零改动

### 2.2 评分函数

v1 使用**方案 A：纯复用计数**（`score = reuse_count`），后续通过消融实验判断是否需要加权混合或时间衰减。

### 2.3 修改文件清单

| 文件 | 修改内容 | 改动量 |
|------|----------|--------|
| `vllm/config/cache.py` | 新增 `EvictionPolicy = Literal["lru", "adaptive"]` + `eviction_policy` 字段 | +9 行 |
| `vllm/engine/arg_utils.py` | 新增 `--eviction-policy` CLI 参数 | +6 行 |
| `vllm/v1/core/kv_cache_utils.py` | `FreeKVCacheBlockQueue` 加 `reuse_counter`、`_popleft_adaptive()`、`increment_reuse()` | +84 行 |
| `vllm/v1/core/block_pool.py` | `touch()` 调用 `increment_reuse()`；驱逐计数器 + cache 利用率 | +31 行 |
| `vllm/v1/core/kv_cache_coordinator.py` | 透传 `eviction_policy`（基类 + 子类 + 工厂函数） | +12 行 |
| `vllm/v1/core/kv_cache_manager.py` | 透传 `eviction_policy` + `prefix_cache_utilization` / `drain_num_evictions` | +9 行 |
| `vllm/v1/core/sched/scheduler.py` | 传递 `eviction_policy` + 填充新 stats 字段 | +4 行 |

**总改动**：7 文件，+155 行（含修改），核心逻辑 ~112 行

### 2.4 配置透传链

```
CLI --eviction-policy
  → EngineArgs.eviction_policy
    → CacheConfig.eviction_policy
      → Scheduler.__init__
        → KVCacheManager(eviction_policy=...)
          → KVCacheCoordinator(eviction_policy=...)
            → BlockPool(eviction_policy=...)
              → FreeKVCacheBlockQueue(eviction_policy=...)
```

### 2.5 任务清单

- [x] 在 `FreeKVCacheBlockQueue` 中增加 `reuse_counter: dict[BlockHashWithGroupId, int]`
- [x] 在 prefix cache 命中路径（`BlockPool.touch()`）中调用 `increment_reuse()` 递增计数
- [x] 修改 `popleft()` → `_popleft_lru()` / `_popleft_adaptive()` 分支
- [x] 添加 `CacheConfig.eviction_policy` + `--eviction-policy` CLI 参数
- [x] 配置透传链：`CacheConfig` → `Scheduler` → `KVCacheManager` → `Coordinator` → `BlockPool` → `FreeKVCacheBlockQueue`
- [x] 单元测试：7 项通过（LRU baseline、unhashed 优先、lowest reuse eviction、popleft_n、空队列等）
- [x] GPU Smoke Test：opt-1.3b + V100 端到端验证通过

---

## 3. 观测系统

### 3.1 已有基础设施（直接复用）

| 组件 | 位置 | 提供的指标 |
|------|------|-----------|
| `PrefixCacheStats` | `vllm/v1/metrics/stats.py` | queries/hits 计数 |
| `CachingMetrics` | `vllm/v1/metrics/stats.py` | 滑动窗口 hit_rate 聚合 |
| `KVCacheMetricsCollector` | `vllm/v1/core/kv_cache_metrics.py` | per-block lifetime、idle time、reuse gaps |
| `FinishedRequestStats` | `vllm/v1/metrics/stats.py` | per-request `num_cached_tokens` |
| Prometheus | `vllm/v1/metrics/loggers.py` | `vllm:prefix_cache_queries/hits`、block histogram |

### 3.2 新增埋点

| 指标 | 实现位置 | 说明 |
|------|----------|------|
| **驱逐计数** | `BlockPool._num_evictions` → `SchedulerStats.num_evictions` → Prometheus `vllm:kv_cache_evictions_total` | 累计驱逐 block 数 |
| **被驱逐 block reuse_count** | `KVCacheEvictionEvent.reuse_count` in `_maybe_evict_cached_block()` | 分析驱逐质量 |
| **Cache 利用率** | `BlockPool.get_prefix_cache_utilization()` → Prometheus `vllm:prefix_cache_utilization` | 有缓存数据的 block 占比 |
| **Per-request 等待时间** | 已有 `FinishedRequestStats.queued_time` | 直接复用 |

### 3.3 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `vllm/v1/core/block_pool.py` | `_num_evictions` 计数器、`drain_num_evictions()`、`get_prefix_cache_utilization()` |
| `vllm/v1/core/kv_cache_metrics.py` | `KVCacheEvictionEvent` 增加 `reuse_count` 字段 |
| `vllm/v1/metrics/stats.py` | `SchedulerStats` 增加 `num_evictions` + `prefix_cache_utilization` |
| `vllm/v1/metrics/loggers.py` | Prometheus `vllm:kv_cache_evictions_total` counter + `vllm:prefix_cache_utilization` gauge |

### 3.4 实验工具脚本

| 脚本 | 用途 |
|------|------|
| `experiments/collect_metrics.py` | 向 vLLM server 发请求，收集 per-request JSONL + Prometheus 汇总 |
| `experiments/plot_comparison.py` | LRU vs Adaptive 对比图（bar chart + latency CDF） |
| `experiments/smoke_test.py` | 端到端 smoke test（启动 server → 发请求 → 验证指标） |

### 3.5 任务清单

- [x] 验证 `PrefixCacheStats` 数据链路完整
- [x] 在 `BlockPool._maybe_evict_cached_block()` 中增加驱逐计数器 + reuse_count 记录
- [x] `SchedulerStats` 增加 `num_evictions` + `prefix_cache_utilization`
- [x] Prometheus 新增 counter + gauge
- [x] 文本日志新增 `Prefix cache util: X%` + `Evictions: N`
- [x] 编写 `experiments/collect_metrics.py`
- [x] 编写 `experiments/plot_comparison.py`
- [x] 端到端验证 Prometheus metrics（102 项含新增）和日志输出正确

---

## 4. Smoke Test 结果（03/29）

**环境**：opt-1.3b + V100 (32GB)，20 条请求（10 唯一 + 10 重复）

| 指标 | LRU | Adaptive | 说明 |
|------|-----|----------|------|
| 请求完成 | ✅ 20/20 | ✅ 20/20 | |
| Cache 查询数 | > 0 | > 0 | |
| 命中率 | 73.4% | 73.4% | 低压力下策略趋同（预期行为） |
| 重复请求加速 | 1.97x | 1.97x | |
| 驱逐数 | 0 | 0 | 20 条请求远未填满 86K token cache |
| Prometheus 指标 | ✅ 102 项 | ✅ 102 项 | 含新增 evictions/utilization |

**结论**：低压力下两种策略表现一致，符合预期。需设计高压 benchmark 触发驱逐才能体现策略差异。

**结果存储**：`experiments/results/2026-03-29_smoke_lru/` 和 `experiments/results/2026-03-29_smoke_adaptive/`

---

## 5. Git 提交历史

| Commit | 说明 |
|--------|------|
| `664deb5ae` | feat: add adaptive eviction policy for prefix cache blocks |
| `9e0ccd81e` | feat: add Prometheus and logging metrics for eviction and cache utilization |
| `31ce2ebaa` | test: add unit tests for adaptive eviction policy |
| `3576435e2` | feat: add experiment scripts and smoke test results |
