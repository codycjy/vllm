# Course Project: Workload-Aware KV Cache Eviction & Cache-Aware Scheduling

> 项目计划索引

## 总体计划

- [主计划](main_plan.md) — 架构分析、策略设计、实验矩阵、时间线

## 子计划

| # | 子计划 | 状态 | 说明 |
|---|--------|------|------|
| 0 | [自适应驱逐 + 观测系统](00_adaptive_eviction_and_observability.md) | ✅ 完成 | GPU 端自适应驱逐策略 + Prometheus/日志观测系统 + Smoke Test |
| 1 | [数据准备](01_data_preparation.md) | ✅ 完成 | 5 个数据集就绪（ShareGPT/MMLU/Burst/Code/RAG） |
| 2 | [缓存感知调度器](02_cache_aware_scheduler.md) | ✅ 完成 | PrefixMatchRequestQueue + aging 公平性机制，18 项测试 + GPU 验证 |
| 3 | [实验验证与执行](03_experiment_validation.md) | 进行中 | 高压驱逐验证 + 自动化实验矩阵 + 分析可视化 |
| 4 | 报告与展示 | 未开始 | 论文 + 幻灯片 + demo |

## 进度总览

| 阶段 | 完成度 | 关键里程碑 |
|------|--------|-----------|
| 代码库分析 | 100% | 驱逐修改点定位、架构理解 |
| 自适应驱逐 | 100% | `--eviction-policy lru\|adaptive`，7 项单元测试，smoke test 通过 |
| 观测系统 | 100% | Prometheus evictions/utilization，text logging，收集脚本 |
| 数据准备 | 100% | ShareGPT 92K、MMLU 14K、Burst/Code/RAG 各 500，脚本可复现 |
| 缓存感知调度 | 100% | `--scheduling-policy prefix_match`，18 项测试，GPU 端到端验证通过 |
| 联合评估 | 0% | |
| 报告 | 0% | |
