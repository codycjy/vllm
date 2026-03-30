# 子计划 2：缓存感知调度器

> 状态：✅ 完成（03/30）
>
> 目标：实现 Prefix-Match 缓存感知请求调度 + Aging 公平性机制，作为实验矩阵中 Scheduling-only 和 Joint 配置的调度策略

---

## 1. 背景与动机

当前 vLLM v1 调度器在处理 waiting 请求时使用 **FCFS**（先来先服务）或 **Priority**（按优先级）策略。两种策略都**不考虑请求与 prefix cache 的匹配程度**。

**问题**：假设 cache 中已缓存某个 system prompt 的 block，此时有两个 waiting 请求：
- 请求 A：先到达，但与 cache 无匹配（需全量计算）
- 请求 B：后到达，但能命中缓存的 system prompt（只需计算后缀）

FCFS 会先调度 A，在计算 A 的过程中，B 对应的 cache block 可能被驱逐。如果先调度 B，不仅 B 更快完成（利用缓存），还能**减少驱逐压力**，间接帮助后续请求。

**缓存感知调度的核心思想**：在 waiting 队列中，优先调度与 prefix cache 匹配度高的请求，从而最大化 cache 利用率。但需加入 **aging 机制**防止低匹配请求饥饿。

---

## 2. 现有代码架构分析

### 2.1 请求队列体系 (`request_queue.py`)

```
SchedulingPolicy(Enum)
  ├── FCFS = "fcfs"
  └── PRIORITY = "priority"

RequestQueue(ABC)                    # 抽象基类，定义 7 个抽象方法
  ├── add_request(request)           # 入队
  ├── pop_request() -> Request       # 按策略弹出
  ├── peek_request() -> Request      # 按策略窥视
  ├── prepend_request(request)       # 放回队首
  ├── prepend_requests(requests)     # 批量放回
  ├── remove_request(request)        # 移除指定
  ├── remove_requests(requests)      # 批量移除
  └── __bool__ / __len__ / __iter__

FCFSRequestQueue(deque, RequestQueue)     # deque 实现 FIFO
PriorityRequestQueue(RequestQueue)        # heapq 实现按 priority 排序

create_request_queue(policy) -> RequestQueue  # 工厂函数
```

### 2.2 调度器 waiting 循环 (`scheduler.py:539-795`)

```python
# 第 535 行：创建临时队列收集被跳过的请求
skipped_waiting_requests = create_request_queue(self.policy)

# 第 539 行：主循环
while self.waiting and token_budget > 0:
    request = self.waiting.peek_request()

    # ... 跳过各种等待状态的请求 ...

    # 第 601-605 行：首次调度时查询 prefix cache
    if request.num_computed_tokens == 0:
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )

    # ... allocate_slots, 如果分配失败则 break ...

    self.waiting.pop_request()
    self.running.append(request)
```

**关键观察**：
1. `get_computed_blocks()` 在**逐个请求处理时**才调用，不是预先批量查询
2. 被跳过的请求通过 `skipped_waiting_requests` 放回队首
3. 循环是**顺序消费** waiting 队列，直到 token budget 用尽或分配失败

### 2.3 配置体系

```python
# scheduler.py:103
SchedulerPolicy = Literal["fcfs", "priority"]

# scheduler.py 构造函数
self.policy = SchedulingPolicy(scheduler_config.policy)
self.waiting = create_request_queue(self.policy)
```

### 2.4 Request 可用信息

| 属性 | 类型 | 说明 |
|------|------|------|
| `request.block_hashes` | `list[BlockHash]` | 请求的 block hash 列表（用于 prefix cache 查找） |
| `request.num_tokens` | `int` | 总 token 数 |
| `request.arrival_time` | `float` | 到达时间戳 |
| `request.priority` | `int` | 优先级（越小越高） |
| `request.num_computed_tokens` | `int` | 已计算的 token 数（新请求为 0） |
| `request.num_preemptions` | `int` | 被抢占次数 |

---

## 3. 实现方案

### 3.1 新增 `PrefixMatchRequestQueue`

在 `request_queue.py` 中新增第三种队列实现，核心思想：**每次 peek/pop 时，对队列中的请求按 prefix cache 命中长度降序排序（加 aging 修正）**。

```python
class PrefixMatchRequestQueue(RequestQueue):
    """
    按 prefix cache 命中长度调度：优先调度与 cache 匹配度高的请求。
    加入 aging 机制防止低匹配请求饥饿。
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        max_wait_seconds: float = 30.0,  # aging 阈值
        rescore_interval: int = 1,       # 每 N 次 peek 重新评分
    ):
        self._queue: deque[Request] = deque()
        self._kv_cache_manager = kv_cache_manager
        self._max_wait_seconds = max_wait_seconds
        self._rescore_interval = rescore_interval
        self._peek_count = 0
        self._sorted = False
```

### 3.2 评分函数设计

```python
def _compute_schedule_score(self, request: Request, now: float) -> float:
    """
    计算请求的调度优先级分数（越高越优先调度）。

    score = prefix_match_ratio + aging_bonus

    - prefix_match_ratio: 命中的 cached token 数 / 总 token 数（0~1）
    - aging_bonus: 等待时间超过阈值后，每超过 1 秒加 0.1 分，确保饥饿请求被提升
    """
    # 查询 prefix cache 命中长度
    _, num_cached_tokens = self._kv_cache_manager.get_computed_blocks(request)
    prefix_match_ratio = num_cached_tokens / max(request.num_tokens, 1)

    # Aging 机制：等待过久的请求获得加分
    wait_time = now - request.arrival_time
    aging_bonus = 0.0
    if wait_time > self._max_wait_seconds:
        aging_bonus = (wait_time - self._max_wait_seconds) * 0.1

    return prefix_match_ratio + aging_bonus
```

### 3.3 排序策略

**不是每次 peek 都全量排序**，而是采用**周期性重排**：

```python
def _maybe_resort(self):
    """按 schedule score 对 waiting 队列重排（周期性）"""
    self._peek_count += 1
    if self._sorted and self._peek_count % self._rescore_interval != 0:
        return

    now = time.monotonic()
    scored = [(self._compute_schedule_score(req, now), i, req)
              for i, req in enumerate(self._queue)]
    scored.sort(key=lambda x: (-x[0], x[1]))  # 分数降序，同分保持原序

    self._queue.clear()
    self._queue.extend(item[2] for item in scored)
    self._sorted = True
```

### 3.4 Aging 机制详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_wait_seconds` | 30.0 | 超过此时间开始获得 aging 加分 |
| aging 增速 | 0.1/秒 | 超过阈值后每秒加 0.1 分 |
| 饥饿保护 | 自动 | 等待 40 秒的请求得到 +1.0 分，必然超过任何 prefix_match_ratio |

**效果**：
- 正常情况下，高 cache 命中的请求被优先调度（prefix_match_ratio 主导）
- 等待过久的请求会被 aging bonus 提升到队列前部（公平性保障）
- 阈值可通过 CLI 参数 `--scheduling-max-wait` 调整

### 3.5 性能开销控制

| 关注点 | 应对措施 |
|--------|----------|
| `get_computed_blocks()` 调用开销 | 只是 hash 查找，O(1) per block，不涉及 GPU 操作 |
| 排序开销 | O(N log N)，N = waiting 队列长度；vLLM 默认 `max_num_seqs=128`，开销可忽略 |
| 频繁重排 | `rescore_interval` 控制频率，默认每次 peek 都重排（N 小时无需优化） |
| 与驱逐策略交互 | `get_computed_blocks()` 是只读查询，不触发驱逐或状态变更 |

---

## 4. 需修改的文件

### 4.1 新增 / 修改文件一览

| 文件 | 修改内容 | 预估改动 |
|------|----------|----------|
| `vllm/v1/core/sched/request_queue.py` | 新增 `PrefixMatchRequestQueue` 类 + `SchedulingPolicy.PREFIX_MATCH` | ~120 行 |
| `vllm/v1/core/sched/scheduler.py` | 创建 waiting 队列时传入 `kv_cache_manager`；新增调度统计 | ~15 行 |
| `vllm/config/scheduler.py` | `SchedulerPolicy` 扩展为 `Literal["fcfs", "priority", "prefix_match"]` | ~5 行 |
| `vllm/engine/arg_utils.py` | 新增 `--scheduling-max-wait` CLI 参数 | ~6 行 |
| `vllm/v1/metrics/stats.py` | `SchedulerStats` 新增 `max_wait_time`、`num_starved_requests` | ~4 行 |
| `vllm/v1/metrics/loggers.py` | Prometheus `vllm:scheduler_max_wait_seconds` gauge | ~10 行 |
| `tests/v1/core/test_prefix_match_scheduler.py` | 单元测试 | ~200 行 |

**总预估**：~360 行新增

### 4.2 详细修改说明

#### 4.2.1 `request_queue.py` — 核心新增

```python
# 扩展枚举
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    PREFIX_MATCH = "prefix_match"     # 新增

# 新增队列类
class PrefixMatchRequestQueue(RequestQueue):
    """按 prefix cache 命中度 + aging 调度"""
    ...

# 工厂函数扩展
def create_request_queue(
    policy: SchedulingPolicy,
    kv_cache_manager=None,       # prefix_match 模式需要
    max_wait_seconds: float = 30.0,
) -> RequestQueue:
    ...
    elif policy == SchedulingPolicy.PREFIX_MATCH:
        assert kv_cache_manager is not None
        return PrefixMatchRequestQueue(kv_cache_manager, max_wait_seconds)
```

#### 4.2.2 `scheduler.py` — 透传 kv_cache_manager

当前 `self.waiting` 在构造函数中创建（~第 130 行附近），此时 `kv_cache_manager` 尚未初始化。需要调整创建时机：

**方案**：延迟创建 waiting 队列，或在 `kv_cache_manager` 初始化后重建。

```python
# 构造函数中（kv_cache_manager 初始化之后）
if self.policy == SchedulingPolicy.PREFIX_MATCH:
    self.waiting = PrefixMatchRequestQueue(
        kv_cache_manager=self.kv_cache_manager,
        max_wait_seconds=self.scheduler_config.scheduling_max_wait,
    )
```

同时在 `schedule()` 的 waiting 循环中，需注意：`get_computed_blocks()` 可能在排序阶段已被调用一次，在后续逐请求处理时会再次调用。由于 `get_computed_blocks()` 是幂等的只读操作，重复调用无副作用，但可考虑缓存结果以避免重复查找。

#### 4.2.3 `scheduler.py` — 调度统计

在 `schedule()` 末尾收集等待时间统计：

```python
# 在 make_stats() 中
if self.waiting:
    now = time.monotonic()
    max_wait = max(now - req.arrival_time for req in self.waiting)
    num_starved = sum(1 for req in self.waiting
                      if now - req.arrival_time > self.scheduling_max_wait)
else:
    max_wait = 0.0
    num_starved = 0
```

#### 4.2.4 `config/scheduler.py` — 配置扩展

```python
SchedulerPolicy = Literal["fcfs", "priority", "prefix_match"]

@dataclass
class SchedulerConfig:
    policy: SchedulerPolicy = "fcfs"
    scheduling_max_wait: float = 30.0  # 新增：aging 阈值（秒）
```

#### 4.2.5 `arg_utils.py` — CLI 参数

```python
parser.add_argument(
    "--scheduling-max-wait",
    type=float,
    default=30.0,
    help="Max wait time (seconds) before aging boost in prefix_match scheduling. "
         "Only effective when --scheduling-policy=prefix_match."
)
```

---

## 5. 与实验矩阵的关系

| 实验配置 | 驱逐策略 | 调度策略 | CLI 参数 |
|---------|----------|---------|----------|
| Baseline | LRU | FCFS | `--eviction-policy lru --scheduling-policy fcfs` |
| Eviction-only | Adaptive | FCFS | `--eviction-policy adaptive --scheduling-policy fcfs` |
| **Scheduling-only** | LRU | **Prefix-match** | `--eviction-policy lru --scheduling-policy prefix_match` |
| **Joint** | Adaptive | **Prefix-match** | `--eviction-policy adaptive --scheduling-policy prefix_match` |

---

## 6. 单元测试计划

| 测试用例 | 验证目标 |
|----------|----------|
| `test_prefix_match_ordering` | 高 cache 命中请求排在低命中请求前面 |
| `test_aging_prevents_starvation` | 等待超过阈值的低命中请求被提升到前面 |
| `test_no_cache_hit_degrades_to_fcfs` | 所有请求命中率为 0 时，行为类似 FCFS |
| `test_prepend_preserves_order` | `prepend_request` 后排序仍正确 |
| `test_remove_request` | 移除请求后队列状态一致 |
| `test_empty_queue_operations` | 空队列的 peek/pop 抛出正确异常 |
| `test_scheduling_max_wait_config` | CLI 参数正确透传到队列 |
| `test_mixed_status_requests` | WAITING_FOR_FSM 等特殊状态请求被正确跳过（由 scheduler 处理，非队列） |

---

## 7. 关键设计决策

### 7.1 排序时机：入队时 vs 出队时

| 方案 | 优点 | 缺点 |
|------|------|------|
| **入队时排序**（heap） | 每次 add O(log N) | cache 状态在入队后会变化，排序失效 |
| **出队时排序**（周期性） ✅ | 反映最新 cache 状态 | 每次 O(N log N)，但 N 小 |

选择**出队时排序**：cache 内容在不断变化（新请求到达 → 新 block 被缓存 → 旧 block 被驱逐），入队时的排名可能很快过时。

### 7.2 `get_computed_blocks()` 调用的安全性

`get_computed_blocks()` 内部调用 `coordinator.find_longest_cache_hit()`，该方法是**纯只读**操作：
- 只做 hash 查找，不分配 block
- 不修改 ref_cnt（不调用 `touch()`）
- 不触发驱逐

因此在排序阶段调用是安全的，不会影响 cache 状态。

### 7.3 与被跳过请求的交互

`scheduler.py` 中有 `skipped_waiting_requests` 机制：某些请求因特殊状态（WAITING_FOR_FSM 等）被跳过后放回队首。`PrefixMatchRequestQueue` 的 `prepend_request()` 需要正确处理：放回后在下次排序时重新参与评分。

### 7.4 Preempted 请求的处理

被抢占的请求通过 `prepend_request()` 放回 waiting 队列前端。这些请求的 `num_computed_tokens` 被重置为 0，会在下次调度时重新查询 prefix cache——如果它们的 prefix 仍在 cache 中，会获得高分；否则正常参与竞争。

---

## 8. 执行步骤

### Phase 1：核心实现（目标 2 天）

- [ ] **Step 1**：扩展 `SchedulingPolicy` 枚举 + `SchedulerConfig` 配置
  - 修改 `vllm/config/scheduler.py`
  - 修改 `vllm/engine/arg_utils.py`

- [ ] **Step 2**：实现 `PrefixMatchRequestQueue`
  - 在 `vllm/v1/core/sched/request_queue.py` 中新增类
  - 实现所有 `RequestQueue` 抽象方法
  - 实现 `_compute_schedule_score()` 评分函数
  - 实现 `_maybe_resort()` 周期性排序

- [ ] **Step 3**：修改 `create_request_queue()` 工厂函数
  - 支持 `PREFIX_MATCH` 分支
  - 接受 `kv_cache_manager` 和 `max_wait_seconds` 参数

- [ ] **Step 4**：修改 `scheduler.py` 透传
  - 在 `kv_cache_manager` 初始化后创建 `PrefixMatchRequestQueue`
  - 确保 `skipped_waiting_requests` 也使用正确的队列类型

### Phase 2：观测与公平性（目标 1 天）

- [ ] **Step 5**：新增调度公平性指标
  - `SchedulerStats` 增加 `max_wait_time`、`num_starved_requests`
  - Prometheus 暴露 `vllm:scheduler_max_wait_seconds` gauge
  - 文本日志新增 `Max wait: Xs, Starved: N`

- [ ] **Step 6**：验证 aging 机制
  - 手动构造长等待场景，确认低匹配请求最终被调度

### Phase 3：测试与验证（目标 1 天）

- [ ] **Step 7**：编写单元测试
  - `tests/v1/core/test_prefix_match_scheduler.py`
  - 覆盖 §6 中的 8 个测试用例

- [ ] **Step 8**：GPU Smoke Test
  - 复用 `experiments/smoke_test.py` 框架
  - 用 opt-1.3b + V100 验证 `--scheduling-policy prefix_match` 功能正确
  - 对比 FCFS vs prefix_match 的调度顺序差异

- [ ] **Step 9**：高压驱逐验证
  - 设计大量唯一前缀的 benchmark，使 cache 填满触发驱逐
  - 验证 prefix_match 调度 + adaptive 驱逐的联合效果

---

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| `get_computed_blocks()` 在排序中被调用 N 次 | N 大时延迟增加 | N = waiting 队列长度，通常 < 128；必要时只对前 K 个评分 |
| 排序改变调度顺序 → token budget 分配不同 | 可能触发不同的抢占行为 | 本质上是期望的效果；通过 aging 保证公平性 |
| `kv_cache_manager` 初始化时序 | waiting 队列创建时 manager 未就绪 | 延迟创建或在 manager 初始化后重建队列 |
| prefix_match 排序开销在高并发时显著 | 调度循环变慢 | `rescore_interval` 控制频率；实际测量后调整 |

---

## 10. 与子计划 0 的衔接

子计划 0 完成了**驱逐侧**的优化（自适应驱逐 + 观测系统），本子计划完成**调度侧**的优化。两者通过以下方式联合工作：

```
请求到达 → PrefixMatchRequestQueue 按 cache 命中度排序
  → 高命中请求优先调度 → 命中的 block 被 touch() → reuse_count++
    → Adaptive 驱逐保护高 reuse 的 block → 更多请求命中
      → 正反馈循环：调度和驱逐协同提升 cache 利用率
```

**独立性**：两者代码路径完全解耦（scheduler 不知道驱逐策略，驱逐不知道调度策略），可独立开发、独立测试、联合评估。
