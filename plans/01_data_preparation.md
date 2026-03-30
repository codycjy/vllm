# 子计划 1：数据准备

> 目标：为 4 配置 × 3 工作负载 × 3 cache budget 实验矩阵准备所有数据集

---

## 1. 数据需求总览

### 1.1 按工作负载类型

| 工作负载 | 数据集 | 前缀共享 | 用途 |
|---------|--------|---------|------|
| **高复用** | ShareGPT（多轮对话） | 高 | 核心实验 |
| **低复用** | MMLU（独立 QA） | 无 | 核心实验 + 负对照 |
| **混合/突发** | 合成突发数据 | 可控 | 核心实验 |
| **补充** | 随机合成 | 无 | vLLM 内置 `--dataset-name random`，无需文件 |

### 1.2 按 Prompt 长度分桶（消融实验）

| 分桶 | Token 范围 | 数据来源 |
|------|-----------|---------|
| 短 | 64–256 | ShareGPT / MMLU 按长度过滤 |
| 中 | 256–1024 | ShareGPT / MMLU 按长度过滤 |
| 长 | 1024–4096 | ShareGPT 长对话 / 合成长前缀 |
| 超长 | 4096–16384 | 合成数据（真实数据在此区间稀少） |

### 1.3 按 Prompt 类型（消融实验）

| 类型 | 数据来源 | 优先级 |
|------|---------|--------|
| 纯文本对话 | ShareGPT | P0 |
| 独立 QA | MMLU | P0 |
| 代码补全 | 合成：共享 repo context | P1 |
| RAG | 合成：Zipfian 文档热度 | P1 |
| Agent/Tool calls | 合成 function-calling | P2 |

---

## 2. 数据集详情与获取方式

### 2.1 ShareGPT — 高复用工作负载（P0）

**来源**: HuggingFace `anon8231489123/ShareGPT_Vicuna_unfiltered`

**格式**: vLLM 原生支持的 ShareGPT JSON
```json
[
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]
```

**获取命令**:
```bash
# 方法 1: huggingface-cli（推荐）
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
  --local-dir /ocean/projects/cis250265p/xli45/opensource/data/sharegpt \
  --repo-type dataset

# 方法 2: 直接下载清洗后版本
wget -O /ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
```

**预处理**:
1. 统计 prompt 长度分布（用 tokenizer 计算 token 数）
2. 按长度分桶导出子集
3. 提取多轮对话子集（≥3 轮）用于高复用场景

**预估大小**: ~500MB（JSON）

**在 benchmark 中使用**:
```bash
# benchmark_prefix_caching.py（专用 prefix cache 测试）
python benchmarks/benchmark_prefix_caching.py \
  --model /ocean/projects/cis250265p/xli45/opensource/models/opt-1.3b \
  --dataset-path /ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --repeat-count 5 \
  --input-length-range "128:1024" \
  --output-len 10 \
  --enable-prefix-caching

# vllm bench serve（通用 serving benchmark）
vllm bench serve \
  --dataset-name sharegpt \
  --dataset-path /ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --sharegpt-output-len 128
```

---

### 2.2 MMLU — 低复用工作负载（P0）

**来源**: HuggingFace `cais/mmlu` 或 `hails/mmlu_no_train`

**特点**: 每个 prompt 独立（选择题），几乎无前缀共享 → 理想负对照

**格式转换**: MMLU 原始格式 → vLLM Custom JSONL
```jsonl
{"prompt": "Question: What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid\nAnswer:", "output_tokens": 10}
```

**获取与转换脚本**: `scripts/prepare_mmlu.py`（需编写）
```bash
# 下载
huggingface-cli download cais/mmlu \
  --local-dir /ocean/projects/cis250265p/xli45/opensource/data/mmlu \
  --repo-type dataset

# 转换为 vLLM custom JSONL 格式
python scripts/prepare_mmlu.py \
  --input-dir /ocean/projects/cis250265p/xli45/opensource/data/mmlu \
  --output /ocean/projects/cis250265p/xli45/opensource/data/mmlu_vllm.jsonl \
  --subjects all \
  --max-prompts 2000
```

**在 benchmark 中使用**:
```bash
vllm bench serve \
  --dataset-name custom \
  --dataset-path /ocean/projects/cis250265p/xli45/opensource/data/mmlu_vllm.jsonl \
  --custom-output-len 10 \
  --num-prompts 500
```

**预估大小**: ~100MB（原始），~20MB（转换后 JSONL）

---

### 2.3 合成突发数据 — 混合/突发工作负载（P0）

**目的**: 模拟真实场景中热前缀 + 冷 prompt 的突发到达模式

**设计**:
- **热前缀**（5 个）：固定的 system prompt，每个 ~200 tokens，高频重复
- **冷 prompt**（无共享）：随机生成的独立 prompt，各不相同
- **突发模式**：时间窗口内 70% 请求命中热前缀，30% 为冷 prompt
- **总请求数**：500–1000

**格式**: Custom JSONL
```jsonl
{"prompt": "[system] You are a helpful assistant...[/system]\nUser: What is...", "output_tokens": 128, "prefix_group": "hot_0"}
{"prompt": "Random unique question about topic XYZ...", "output_tokens": 128, "prefix_group": "cold"}
```

**生成脚本**: `scripts/generate_burst_data.py`（需编写）

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-hot-prefixes` | 5 | 热前缀数量 |
| `--hot-prefix-len` | 200 | 热前缀 token 长度 |
| `--suffix-len` | 50–200 | 每个请求的独特后缀长度（随机范围） |
| `--num-requests` | 500 | 总请求数 |
| `--hot-ratio` | 0.7 | 热前缀请求占比 |
| `--burst-size` | 20 | 每个突发窗口内的请求数 |
| `--output-tokens` | 128 | 生成 token 数 |

**在 benchmark 中使用**:
```bash
vllm bench serve \
  --dataset-name custom \
  --dataset-path /ocean/projects/cis250265p/xli45/opensource/data/burst_synthetic.jsonl \
  --custom-output-len 128 \
  --num-prompts 500
```

---

### 2.4 随机合成 — 纯负对照（P0）

**无需文件**，vLLM 内置：
```bash
vllm bench serve \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 512 \
  --random-output-len 128 \
  --random-range-ratio 0.3 \
  --random-prefix-len 0
```

---

### 2.5 合成代码补全 — Prompt 类型消融（P1）

**设计**: 模拟多个用户在同一 repo 上做代码补全
- **共享前缀**: 固定的 repo context（文件头 + import 语句），~500 tokens
- **独特后缀**: 不同的函数体/补全位置
- **前缀组数**: 3–5 个"文件"

**格式**: Custom JSONL
**生成脚本**: `scripts/generate_code_completion.py`（需编写）

---

### 2.6 合成 RAG — Prompt 类型消融（P1）

**设计**: 模拟热门文档被多次查询
- **文档池**: 20 篇文档，热度服从 Zipfian 分布
- **前缀**: "Based on the following document: [document text]\n\nQuestion: "
- **后缀**: 不同的查询问题

**格式**: Custom JSONL
**生成脚本**: `scripts/generate_rag_data.py`（需编写）

---

## 3. 存储规划

```
/ocean/projects/cis250265p/xli45/opensource/data/
├── sharegpt/
│   ├── ShareGPT_V3_unfiltered_cleaned_split.json    # ~500MB 原始
│   ├── sharegpt_short.json                           # 64-256 tokens
│   ├── sharegpt_medium.json                          # 256-1024 tokens
│   └── sharegpt_long.json                            # 1024-4096 tokens
├── mmlu/
│   └── (HuggingFace 原始)                            # ~100MB
├── mmlu_vllm.jsonl                                   # 转换后 ~20MB
├── burst_synthetic.jsonl                             # ~5MB
├── code_completion_synthetic.jsonl                   # ~5MB
└── rag_synthetic.jsonl                               # ~10MB
```

**总存储**: ~700MB

---

## 4. 需要编写的脚本

| 脚本 | 输入 | 输出 | 优先级 | 状态 |
|------|------|------|--------|------|
| `scripts/prepare_mmlu.py` | HF mmlu 数据集 | `mmlu_vllm.jsonl` | P0 | ✅ |
| `scripts/generate_burst_data.py` | 参数配置 | `burst_synthetic.jsonl` | P0 | ✅ |
| `scripts/analyze_dataset.py` | 任意数据集 + tokenizer | 长度分布统计 + 分桶导出 | P1 | 暂缓（vLLM 自带 `--input-length-range` 过滤） |
| `scripts/generate_code_completion.py` | 参数配置 | `code_completion_synthetic.jsonl` | P1 | ✅ |
| `scripts/generate_rag_data.py` | 参数配置 | `rag_synthetic.jsonl` | P1 | ✅ |

---

## 5. 执行顺序

### Phase 1: 核心实验数据（P0）— 目标 04/05

- [x] **Step 1**: 下载 ShareGPT 数据集
  ```bash
  mkdir -p /ocean/projects/cis250265p/xli45/opensource/data/sharegpt
  wget -O /ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
  ```

- [ ] **Step 2**: （暂缓）编写 `scripts/analyze_dataset.py` — vLLM 自带长度过滤

- [ ] **Step 3**: （暂缓）分析 ShareGPT 长度分布，导出分桶子集
  ```bash
  python scripts/analyze_dataset.py \
    --input /ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
    --format sharegpt \
    --tokenizer /ocean/projects/cis250265p/xli45/opensource/models/opt-1.3b \
    --output-dir /ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ \
    --buckets "64:256,256:1024,1024:4096"
  ```

- [x] **Step 4**: 下载 MMLU 并转换格式
  ```bash
  huggingface-cli download cais/mmlu \
    --local-dir /ocean/projects/cis250265p/xli45/opensource/data/mmlu \
    --repo-type dataset
  python scripts/prepare_mmlu.py \
    --input-dir /ocean/projects/cis250265p/xli45/opensource/data/mmlu \
    --output /ocean/projects/cis250265p/xli45/opensource/data/mmlu_vllm.jsonl
  ```

- [x] **Step 5**: 编写并运行合成突发数据生成
  ```bash
  python scripts/generate_burst_data.py \
    --num-hot-prefixes 5 \
    --hot-prefix-len 200 \
    --num-requests 500 \
    --hot-ratio 0.7 \
    --output /ocean/projects/cis250265p/xli45/opensource/data/burst_synthetic.jsonl
  ```

- [x] **Step 6**: 验证所有数据集格式正确（字段检查通过）
  ```bash
  # 快速验证（dry run，只检查格式不跑推理）
  python -c "
  import json
  data = json.load(open('/ocean/projects/cis250265p/xli45/opensource/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'))
  print(f'ShareGPT: {len(data)} conversations')
  assert all('conversations' in d for d in data[:100])
  print('ShareGPT format OK')
  "
  ```

### Phase 2: 消融实验数据（P1）— 目标 04/10

- [x] **Step 7**: 生成代码补全合成数据（500 请求，5 文件上下文）
- [x] **Step 8**: 生成 RAG 合成数据（500 请求，20 文档，Zipfian）
- [ ] **Step 9**: 生成超长 prompt 合成数据（4096–16384 tokens）— 按需

---

## 6. 与实验矩阵的对应关系

| 实验矩阵维度 | 数据集 | benchmark 命令关键参数 |
|-------------|--------|----------------------|
| 高复用工作负载 | ShareGPT | `--dataset-name sharegpt --dataset-path .../ShareGPT_V3.json` |
| 低复用工作负载 | MMLU | `--dataset-name custom --dataset-path .../mmlu_vllm.jsonl` |
| 混合/突发工作负载 | 合成突发 | `--dataset-name custom --dataset-path .../burst_synthetic.jsonl` |
| 短 prompt 消融 | ShareGPT 分桶 | `--input-length-range "64:256"` 或分桶子集 |
| 中 prompt 消融 | ShareGPT 分桶 | `--input-length-range "256:1024"` |
| 长 prompt 消融 | ShareGPT 分桶 | `--input-length-range "1024:4096"` |
| 代码补全消融 | 合成代码 | `--dataset-name custom --dataset-path .../code_completion.jsonl` |
| RAG 消融 | 合成 RAG | `--dataset-name custom --dataset-path .../rag_synthetic.jsonl` |
| 负对照 | Random | `--dataset-name random --random-prefix-len 0` |

---

## 7. 注意事项

1. **Tokenizer 一致性**: 分析长度时必须用实验模型的 tokenizer（opt-1.3b 或 Qwen3-8B），不同 tokenizer 的 token 数不同
2. **vLLM 过滤**: vLLM benchmark 默认过滤 prompt_len < 4 或 prompt_len + output_len > 2048 的请求，注意调整 `--max-total-len` 参数
3. **可复现性**: 所有脚本使用 `--seed 42`，记录完整命令到实验日志
4. **存储**: 所有数据存放在 `/ocean/projects/cis250265p/xli45/opensource/data/`，与代码分离
