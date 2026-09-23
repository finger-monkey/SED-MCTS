# mcts_prior_experiments

本文件夹是从原工程中**整理出来的最小可复现实验项目**，用于复现两个对照实验：

- **实验 1**：MCTS rollout A/B（Random vs Guided）
- **实验 2**：先验 Bootstrap 网格消融（bootstrap_random_expressions × bootstrap_topk）

## 目录结构（关键文件）

- `ab_compare_mcts_rollout.py`：实验 1 一键脚本
- `run_prior_bootstrap_ablation.py`：实验 2 一键脚本（输出 CSV + 热力图）
- `1D_Advection_Sols_beta1.0_Num200.hdf5`：1D Advection 数据
- `MCTS-4-SR/`：符号回归求解器代码（作为本项目的本地依赖）
- `requirements.txt`：本项目依赖（包含 `MCTS-4-SR/requirements.txt`）

## 环境准备

建议 Python 3.10+。

在本目录下执行：

```bash
python -m pip install -r requirements.txt
```

## 复现实验

确保你的当前工作目录就是本项目根目录（也就是这个 `README.md` 所在目录），然后运行：

### 实验 1：Rollout A/B

```bash
python ab_compare_mcts_rollout.py --seeds 0 1 2 --max_expressions 3000 --bootstrap_random_expressions 2000 --bootstrap_topk 120
```

### 实验 2：先验 Bootstrap 消融

```bash
python run_prior_bootstrap_ablation.py --seeds 0 1 2 --max_expressions 1200 --bootstrap_random_expressions_grid 500 1000 2000 4000 --bootstrap_topk_grid 25 50 100 200
```

输出会写到 `ablation_outputs/` 下（CSV + PNG 热力图）。

## 常见问题

- 如果提示找不到数据文件：确认 `1D_Advection_Sols_beta1.0_Num200.hdf5` 位于项目内（本目录或其子目录），并从项目根目录运行脚本。
- 如果安装 `nlopt` 失败：这通常与本机 Python/编译环境有关，可先用 conda 安装或换用对应的预编译 wheel（不同机器情况不一）。

