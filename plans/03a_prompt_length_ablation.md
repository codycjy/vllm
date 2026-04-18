# 子计划 4：Prompt 长度消融实验

> 状态：📋 计划中（待执行）
>
> 目标：验证"prompt 越长，自适应驱逐和缓存感知调度的收益越大"这一核心假设
>
> 预计 GPU 时间：~2–3 小时（V100）

---

## 1. 实验目标

**核心问题**：不同 prompt 长度下，4 种策略配置的性能差异如何？

**假设**：
- 短 prompt（≤256 token）→ cache 压力低 → 4 种策略几乎无差异
- 长 prompt（≥512 token）→ cache 填满更快 → Adaptive 驱逐保护热前缀的优势显现
- 超长 prompt（≥1024 token）→ 极高 cache 压力 → 联合优化（Joint）效果最明显

---

## 2. 实验设计

### 2.1 长度分桶

| Bucket | Token 目标范围 | 生成词数 | 文件名 |
|--------|---------------|---------|--------|
| short  | 64–256 tokens | 100 词  | `bucket_short.json` |
| medium | 256–512 tokens | 320 词  | `bucket_medium.json` |
| long   | 512–1024 tokens | 700 词  | `bucket_long.json` |
| xlarge | 1024–1800 tokens | 1300 词 | `bucket_xlarge.json` |

每个 bucket：**20 个唯一 prompt × 重复 10 次 = 200 条请求**
- 20 个 prompt 共享同一个长前缀（模拟 system prompt 或文档前缀）
- 每个 prompt 的结尾有不同的问题编号（保证唯一性）
- 200 条请求中 prefix caching 命中率理论上最高可达 90%（第 2–10 次重复均命中）

### 2.2 策略配置

| # | 配置名 | eviction-policy | scheduling-policy |
|---|--------|-----------------|------------------|
| 1 | baseline | lru | fcfs |
| 2 | eviction | adaptive | fcfs |
| 3 | scheduling | lru | prefix_match |
| 4 | joint | adaptive | prefix_match |

**共 4 × 4 = 16 次运行**

### 2.3 Cache 压力设计（精确计算）

模型：**Qwen3-8B**（主评估模型，~16GB on V100 32GB）
使用 `--num-gpu-blocks-override 256`（256 × 16 = **4096 token** 缓存上限）

**驱逐触发阈值 = 256 blocks ÷ 20 unique × 16 tokens/block = 205 tokens/prompt**

| Bucket | 估算 tokens | full blocks/prompt | 20 unique 总需 | 缓存压力 | 驱逐? |
|--------|------------|-------------------|---------------|---------|------|
| short  | ~141 tok   | 8 blocks          | 160 blocks    | 0.6x    | ❌ 不驱逐（负对照）|
| medium | ~429 tok   | 26 blocks         | 520 blocks    | 2.0x    | ✅ 驱逐 |
| long   | ~919 tok   | 57 blocks         | 1140 blocks   | 4.5x    | ✅ 高压 |
| xlarge | ~1699 tok  | 106 blocks        | 2120 blocks   | 8.3x    | ✅ 极高压 |

### 2.4 Scheduling 并发要求（重要！）

`prefix_match` 调度需要**多个请求同时在等待队列中**才有意义——调度器必须有"选哪个"的机会。

| 配置 | collect_metrics 并发数 | 原因 |
|------|----------------------|------|
| baseline, eviction | `--concurrency 1`（串行）| 驱逐测试不需要并发 |
| scheduling, joint  | `--concurrency 10`（并发）| 让调度器有 10 个请求可选 |

`run_prompt_length_ablation.sh` 已自动处理这一差异。

---

## 3. 文件结构

```
scripts/
├── prepare_length_buckets.py    # [新建] 生成 4 个分桶数据集
├── run_prompt_length_ablation.sh # [新建] 跑完全部 16 组实验
└── analyze_prompt_length.py     # [新建] 汇总结果并出图

/ocean/projects/cis250265p/xli45/opensource/data/prompt_length/
├── bucket_short.json            # 200 条请求（short bucket）
├── bucket_medium.json
├── bucket_long.json
└── bucket_xlarge.json

experiments/results/prompt_length_ablation/
├── short_baseline/metrics.jsonl + metrics.summary.json
├── short_eviction/...
├── short_scheduling/...
├── short_joint/...
├── medium_*/...
├── long_*/...
└── xlarge_*/...
```

---

## 4. 环境 Setup

### 4.1 节点申请（你负责）

```bash
# 在 Bridges-2 上申请 V100 交互式节点
interact -p GPU-shared --gres=gpu:v100-32:1 -t 04:00:00
```

等分配到 GPU 节点后再执行后续步骤。

### 4.2 Step 1：生成数据（登录节点即可，无 GPU）

```bash
cd /ocean/projects/cis250265p/xli45/opensource/dev/vllm

# 检查输出目录
mkdir -p /ocean/projects/cis250265p/xli45/opensource/data/prompt_length

# 运行数据生成脚本（约 10 秒）
python3 scripts/prepare_length_buckets.py \
    --output-dir /ocean/projects/cis250265p/xli45/opensource/data/prompt_length

# 验证输出
for f in short medium long xlarge; do
    echo -n "bucket_${f}.json: "
    python3 -c "import json; d=json.load(open('/ocean/projects/cis250265p/xli45/opensource/data/prompt_length/bucket_${f}.json')); print(f'{len(d)} entries, first prompt chars: {len(d[0][\"conversations\"][0][\"value\"])}')"
done
```

### 4.3 Step 2：运行实验（需要 GPU 节点）

```bash
# 在 GPU 节点上运行（确认 CUDA 可用）
nvidia-smi

# 赋予脚本执行权限
chmod +x /ocean/projects/cis250265p/xli45/opensource/dev/vllm/scripts/run_prompt_length_ablation.sh

# 运行全部 16 组实验（预计 2–3 小时）
/ocean/projects/cis250265p/xli45/opensource/dev/vllm/scripts/run_prompt_length_ablation.sh
```

**单独运行某一配置**（调试用）：
```bash
# 只跑 medium bucket 的 baseline 配置
ONLY_BUCKET=medium ONLY_CONFIG=baseline \
    /ocean/projects/cis250265p/xli45/opensource/dev/vllm/scripts/run_prompt_length_ablation.sh
```

### 4.4 Step 3：分析结果（登录节点即可）

```bash
# 生成汇总图表
python3 scripts/analyze_prompt_length.py \
    --results-dir experiments/results/prompt_length_ablation \
    --output experiments/results/prompt_length_ablation/plots

# 查看图表
ls experiments/results/prompt_length_ablation/plots/
```

---

## 5. 关键指标与预期结果

| 指标 | 来源 | 预期趋势 |
|------|------|---------|
| **cache_hit_rate** | Prometheus delta | short≈1.0（所有策略），xlarge: joint > scheduling > eviction > baseline |
| **avg_latency_ms** | per-request JSONL | short：无差别；xlarge：joint 最低 |
| **eviction_count** | Prometheus delta | short≈0；xlarge：adaptive < lru |
| **speedup_vs_baseline** | 衍生 | 随 prompt 长度单调增加（对 joint 配置）|

---

## 6. 时间估算

| 步骤 | 时间 | 硬件 |
|------|------|------|
| 生成数据（prepare_length_buckets.py） | ~1 分钟 | 登录节点 |
| 每次实验运行（start server + 200 req + stop） | ~8–12 分钟 | V100 |
| 全部 16 次运行 | ~2–3 小时 | V100 |
| 分析出图（analyze_prompt_length.py） | ~2 分钟 | 登录节点 |
| **总计（GPU 时间）** | **~2–3 小时** | V100 |

---

## 7. Troubleshooting

| 问题 | 原因 | 解决 |
|------|------|------|
| Server 启动超时 | 模型加载慢，CUDA 初始化 | 增大 `SERVER_WAIT_TIMEOUT=180` 环境变量 |
| `model not found` 错误 | collect_metrics.py 用的 model name "default" | 确认 `--served-model-name default` 已设置 |
| OOM 错误 | xlarge bucket + num-gpu-blocks-override 太大 | 降低 `--num-gpu-blocks-override` 到 128 |
| prefix_match 调度下请求超时 | aging 等待时间太长 | 降低 `--scheduling-max-wait` 到 10 |
| Port 已被占用 | 上一次 server 未退出 | `pkill -f "vllm.entrypoints.openai"` |
