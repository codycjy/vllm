# 子计划 3a：Prompt 长度消融实验

> 状态：🔄 v2 进行中（workload 从 uniform → skewed popularity 重做）
>
> 目标：**回答 TA Q3** — "What's the relationship between the prompt lengths and the performance improvement?"
>
> 负责人：xli45（团队中的 length 维度）；其他维度由队友负责：
>   - 队友 A：prompt 类型维度（TA Q4）
>   - 队友 B：workload × cache budget 实验矩阵

---

## 1. 实验目标

**TA Q3 直译**：我们提出的系统（adaptive eviction + cache-aware scheduling）相对 baseline 的 performance improvement，在 prompt 长度维度上如何变化？

**交付物**：一张清晰的 "improvement vs length" 曲线，给出以下形式的结论：
> "Our proposed system delivers up to **+X pp** hit rate improvement / **Y% latency speedup** in the [medium] length regime. The benefit [grows / peaks / plateaus] with length because [机制解释：cache pressure, scheduling queue depth, block granularity]."

---

## 2. v1 → v2 方法论演进（重要）

### v1 的 workload 问题（2026-04-17 发现）

原 workload（uniform popularity）：
- 20 unique prompts，每个出现 **10 次**（均匀分布）
- 每个 prompt 有独立的随机前缀，无共享
- 固定长度一桶

**观察到的问题**：
1. `scheme_b` vs `joint_b` 几乎没差别 → scheduling 对 joint 没贡献
2. `sched_only` vs `baseline` 也几乎没差别 → scheduling 独立效果微弱
3. 整体故事变成"只有 eviction 有效，scheduling 和 joint 是摆设"，违背 team proposal 的联合优化主张

**根因**：workload 既不激活 eviction 的异质性信号（reuse_count 均匀），也不给 scheduling 创造有意义的等待队列异质性（所有请求 cache-hit 概率相同）。

- **Adaptive eviction 需要**：异质 reuse frequency（热前缀 vs 冷前缀）
- **Prefix_match scheduling 需要**：深等待队列 + 异质 cache-hit 概率

均匀 popularity 两者都不提供。

### v2 workload 设计：skewed popularity

4 个 hot prompt × 30 次 + 16 个 cold prompt × 5 次 = 200 requests（规模不变）。

| Workload 属性 | v1 (uniform) | v2 (skewed) |
|---|---|---|
| Prompt 分布 | 20 × 10 次 | 4 × 30 次 + 16 × 5 次 |
| Reuse 频率异质性 | ❌ 均匀 | ✅ 6:1 |
| Cache-hit 概率异质性 | ❌ | ✅ Hot prompt prefix 常驻 |
| 总请求数 | 200 | 200 |

**模拟现实**：hot prompt = 常见的 system prompt / 高频查询（如 "请总结以下文档"），cold prompt = one-off 查询。这符合 proposal §4.1 "distinguish high-reuse prefixes from one-off prompts" 的目标场景。

### v1 数据保留

v1 数据已提交到 git（`checkpoint: uniform-popularity length ablation + alpha/cache sweeps`），并移动到 `experiments/results/prompt_length_ablation_old/` 作为：
1. **方法论对比**：报告里用作 "uniform vs skewed" 的 control，说明 workload 选择的必要性
2. **回退点**：v2 数据异常时可以对照检查

---

## 3. v2 实验设计

### 3.1 长度分桶（与 v1 一致）

| Bucket | Token 目标范围 | 前缀词数 | 生成块数 (block_size=16) |
|--------|---------------|---------|--------------------------|
| short  | 64–256 tokens | 100 词  | ~8 blocks/prompt |
| medium | 256–512 tokens | 320 词  | ~26 blocks/prompt |
| long   | 512–1024 tokens | 700 词  | ~57 blocks/prompt |
| xlarge | 1024–1800 tokens | 1300 词 | ~106 blocks/prompt |

### 3.2 Skewed popularity 参数

| Prompt 类别 | 数量 | 每个重复次数 | 总请求 | 占比 |
|---|---|---|---|---|
| Hot (高复用) | 4 | 30 | 120 | 60% |
| Cold (one-off) | 16 | 5 | 80 | 40% |
| **合计** | 20 | — | 200 | 100% |

6:1 的重复次数比提供足够的 reuse_count 异质性区分 hot 和 cold block。

### 3.3 策略配置（与 v1 一致）

| # | 配置名 | eviction-policy | scheduling-policy | 目的 |
|---|--------|-----------------|------------------|------|
| 1 | baseline | lru | fcfs | 对照（原生 vLLM）|
| 2 | sched_only | lru | prefix_match | 纯调度贡献 |
| 3 | adaptive_a1.0 | adaptive (α=1.0) | fcfs | Scheme A：纯复用频率 |
| 4 | adaptive_a0.7 | adaptive (α=0.7) | fcfs | Scheme B：70% 频率 + 30% recency |
| 5 | joint_a1.0 | adaptive (α=1.0) | prefix_match | Scheme A 联合调度 |
| 6 | joint_a0.7 | adaptive (α=0.7) | prefix_match | Scheme B 联合调度 |

6 config × 4 bucket = **24 runs**

### 3.4 Cache 压力设计

模型：**Qwen3-8B**，block_size=16
`--num-gpu-blocks-override 256`（4096 tokens cache）

| Bucket | blocks/prompt | 20 unique 总需 | 压力 | 预期行为 |
|--------|---------------|---------------|------|---------|
| short  | ~8   | 160 blocks  | 0.6× | 不驱逐（低压对照）|
| medium | ~26  | 520 blocks  | 2.0× | 🎯 sweet spot 预期 |
| long   | ~57  | 1140 blocks | 4.5× | 高压区 |
| xlarge | ~106 | 2120 blocks | 8.3× | 极高压 |

### 3.5 Skewed 下的新假设（预期结果）

| 策略 | v1 结果（uniform） | v2 预期（skewed） | 机制 |
|---|---|---|---|
| `adaptive` | medium 桶 +9.6pp，其他微弱 | 更多桶显著改进 | reuse_count 差距 6:1，热前缀被稳定保护 |
| `sched_only` | 近似 baseline | long/xlarge 桶 +3-5pp | 深队列下能优先 cached 的 hot 请求 |
| `joint` | ≈ adaptive alone | long 桶放大效应最明显 | eviction 保 hot → scheduler 优选 hot → 相互增强 |

### 3.6 Client 并发

固定 `concurrency=8`。注意 short/medium 桶下 8 个请求可能全部 running，waiting queue 为 0，scheduling 无效 —— 这是 v2 仍存在的局限（记入 limitation 章节）。

---

## 4. 执行步骤

### 4.1 环境准备

```bash
# 在 Bridges-2 上申请 V100 交互式节点
interact -p GPU-shared --gres=gpu:v100-32:1 -t 04:00:00
nvidia-smi   # 确认 GPU 可用
```

### 4.2 Step 1：备份 v1 数据（已完成）

已在 plan v2 启动前将：
- `experiments/results/prompt_length_ablation/` → `experiments/results/prompt_length_ablation_old/`
- `experiments/results/cache_size_sweep/` → `experiments/results/cache_size_sweep_old/`
- `/ocean/projects/cis250265p/xli45/opensource/data/prompt_length/*.json` → `/ocean/projects/cis250265p/xli45/opensource/data/prompt_length/old/`

### 4.3 Step 2：生成 v2 skewed 数据（登录节点即可）

```bash
cd /ocean/projects/cis250265p/xli45/opensource/dev/vllm

python3 scripts/prepare_length_buckets.py \
    --popularity skewed \
    --output-dir /ocean/projects/cis250265p/xli45/opensource/data/prompt_length

# 验证：每个 bucket 应有 200 条，其中 120 条来自 4 个 hot prompt，80 条来自 16 个 cold prompt
```

### 4.4 Step 3：运行 v2 完整矩阵（GPU 节点）

```bash
# 24 runs × ~7 分钟 ≈ 3 小时
./scripts/run_prompt_length_ablation.sh
```

### 4.5 Step 4：分析结果（登录节点）

```bash
singularity exec /ocean/projects/cis250265p/xli45/opensource/containers/images/vllm.sif \
    python3 scripts/analyze_full_ablation.py

# 出图到 experiments/results/prompt_length_ablation/plots/
```

---

## 5. 关键指标

| 指标 | 来源 | 预期趋势（v2 skewed） |
|------|------|---------|
| **cache_hit_rate** | Prometheus delta | short 接近饱和；medium-long 各策略显著分化 |
| **avg_latency_ms** | per-request JSONL | 与 hit rate 反相关 |
| **avg_ttft_ms**（新）| streaming 响应 | cache 命中直接降低 prefill 时间 |
| **eviction_count** | Prometheus delta | adaptive < lru 应显著，尤其 long 桶 |
| **improvement vs baseline** | 衍生 | joint 在 medium/long 有 peak |

---

## 6. 预期报告 section 结构

```
§ Prompt Length Ablation (答 TA Q3)

§.1 Setup
   - Skewed-popularity synthetic workload
   - 4 length buckets × 6 configs = 24 runs
   - Qwen3-8B on V100, cache=256 blocks

§.2 Main Result: Improvement vs Length
   - Primary figure: (joint vs baseline) gain curve across buckets
   - Key finding: sweet spot at [medium/long], max +X pp hit rate

§.3 Component Attribution (ablation within ablation)
   - Decompose improvement: eviction share vs scheduling share
   - Alpha sensitivity (leverage existing α sweep data)

§.4 Cache Sensitivity (机制)
   - Cache-size sweep on medium bucket (已有数据)
   - 2x pressure is the hit-rate sweet spot
   - Xlarge behaves differently due to block granularity

§.5 Workload Sensitivity (limitation / methodology)
   - v1 uniform vs v2 skewed comparison
   - "Without popularity skew, the heterogeneity signal adaptive and
      prefix-match rely on is absent; strategy differences vanish.
      We chose skewed popularity to properly exercise the proposed system."
```

---

## 7. 时间估算

| 步骤 | 时间 | 硬件 |
|------|------|------|
| 备份 v1 数据 | 1 分钟 | 登录节点 |
| 改 prepare_length_buckets.py 加 --popularity flag | 15 分钟 | 登录节点 |
| 生成 v2 数据 | 10 秒 | 登录节点 |
| 跑 24 runs | ~3 小时 | V100 |
| 分析 + 出图 | 5 分钟 | 登录节点 |
| 写报告 section | ~4 小时 | 登录节点 |
| **总计（GPU 时间）** | **~3 小时** | V100 |

---

## 8. 已有可复用成果

| 资产 | 位置 | 用途 |
|------|------|------|
| Scheme B 实现 | `vllm/v1/core/kv_cache_utils.py::_popleft_adaptive` | α 加权驱逐，已落地 |
| α sweep 结果 | `prompt_length_ablation_old/` 中 `*_adaptive_a{0.3,0.5,0.7,0.9,1.0}/` | 论文 §3.3 α sensitivity |
| Cache sweep 结果 | `cache_size_sweep_old/` | 论文 §.4 机制解释 |
| Run/analyze 脚本 | `scripts/run_prompt_length_ablation.sh`, `analyze_full_ablation.py` | v2 直接复用 |
| 命名规范 | `{bucket}_{policy}_a{alpha}` | 统一磁盘结构 |

---

## 9. Troubleshooting

| 问题 | 原因 | 解决 |
|------|------|------|
| Server 启动超时 | 模型加载慢 | 增大 `SERVER_WAIT_TIMEOUT=180` |
| `model not found` | collect_metrics 用 "default" 别名 | 确认 `--served-model-name default` 设置 |
| OOM | xlarge + concurrency=8 + 长 output | 降 `--num-gpu-blocks-override` 或 `concurrency` |
| Orphan server 占 GPU | wrapper kill 不级联 | 脚本已用 `pkill -f vllm.entrypoints.openai.api_server` 兜底 |
| Scheduling 在短桶无效果 | 无 waiting queue | 预期行为，记入 limitation |

---

## 10. 与其他子计划的关系

```
队友 A (类型维度，TA Q4) ────┐
                              ├──→ 团队报告
本子计划 (长度维度，TA Q3) ──┤
                              │
队友 B (实验矩阵) ───────────┘
```

三人数据交汇于报告的 "Evaluation" 章节，分别沿 type × length × workload-cache-budget 三个维度覆盖，形成完整的性能 characterization。
