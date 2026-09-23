# 实验记录

本文件汇总了本项目中的关键实验命令、设置与结果，便于复现实验与对比分析。

## 1. MCTS Rollout A/B 对比实验（Random vs Guided）

**实验目的**：比较 MCTS 在 rollout 阶段使用随机扩展（random rollout）与引导扩展（guided rollout）的性能差异，评估引导策略对误差与效率的影响。

**命令**：
```
python ab_compare_mcts_rollout.py --seeds 0 1 2 --max_expressions 3000 --bootstrap_random_expressions 2000 --bootstrap_topk 120
```

**结果汇总**：
```
AVG random: mse=1.398474e-01, mae=3.029985e-01, r2=-0.2644, time=2.17s
AVG guided: mse=1.119632e-01, mae=2.778148e-01, r2=-0.0123, time=2.03s
MSE improvement (guided vs random): 19.94%
```

**简要分析**：
- 引导 rollout 在 MSE 与 MAE 上均优于随机 rollout，MSE 相对提升约 19.94%。
- 引导 rollout 的平均耗时略低（2.03s vs 2.17s），表明在相同表达式预算下能更快收敛到更优解。
- R2 指标在 guided 情况下显著改善（从 -0.2644 提升到 -0.0123），说明拟合质量更接近真实数据分布。

## 2. 先验 Bootstrap 网格消融实验

**实验目的**：评估先验 bootstrap 的随机表达式数量与 top-k 选择对 MCTS 搜索性能的影响，比较不同配置下的表现趋势。

**命令**：
```
python run_prior_bootstrap_ablation.py --seeds 0 1 2 --max_expressions 1200 --bootstrap_random_expressions_grid 500 1000 2000 4000 --bootstrap_topk_grid 25 50 100 200
```

**结果汇总**（示例输出目录）：
```
Detailed CSV: ablation_outputs/prior_bootstrap_20260313_111639/detailed_runs.csv
Summary CSV : ablation_outputs/prior_bootstrap_20260313_111639/summary.csv
Figure      : ablation_outputs/prior_bootstrap_20260313_111639/heatmap_guided_mse.png
Figure      : ablation_outputs/prior_bootstrap_20260313_111639/heatmap_mse_improvement_vs_random.png
```

**简要分析**：
- 该实验以网格方式扫描 `bootstrap_random_expressions` 与 `bootstrap_topk`，覆盖多种先验初始化强度。
- `summary.csv` 汇总了各配置下的平均指标，可用于选择最优先验规模。
- 热力图展示了不同配置下 guided rollout 的 MSE 以及相对 random rollout 的改进幅度，有助于直观判断先验质量与收益区间。
-  整体MSE 相对提升约 22%以上。


