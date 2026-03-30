# Course Project: Workload-Aware KV Cache Eviction & Cache-Aware Scheduling

> 项目计划索引

## 总体计划

- [主计划](main_plan.md) — 架构分析、策略设计、实验矩阵、时间线

## 子计划

| # | 子计划 | 状态 | 说明 |
|---|--------|------|------|
| 1 | [数据准备](01_data_preparation.md) | ✅ 完成 | 5 个数据集就绪（ShareGPT/MMLU/Burst/Code/RAG） |
| 2 | 缓存感知调度器 | 未开始 | PrefixMatchRequestQueue + aging 机制 |
| 3 | 实验执行 | 未开始 | 自动化运行脚本 + 完整矩阵 |
| 4 | 分析与可视化 | 未开始 | 结果分析 + 图表生成 |
| 5 | 报告与展示 | 未开始 | 论文 + 幻灯片 + demo |

## 进度总览

| 阶段 | 完成度 | 关键里程碑 |
|------|--------|-----------|
| 代码库分析 | 100% | 驱逐修改点定位、架构理解 |
| 自适应驱逐 | 100% | `--eviction-policy lru\|adaptive`，7 项单元测试，smoke test 通过 |
| 观测系统 | 100% | Prometheus evictions/utilization，text logging，收集脚本 |
| 数据准备 | 100% | ShareGPT 92K、MMLU 14K、Burst/Code/RAG 各 500，脚本可复现 |
| 缓存感知调度 | 0% | |
| 联合评估 | 0% | |
| 报告 | 0% | |
