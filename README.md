## Runtime Environment


Run the following commands in this directory:

```
python -m pip install -r .\requirements.txt
python -m pip install -r ".\ablation experiment\requirements.txt"
```

## Run Commands

```
python .\run_demo.py


python .\run_sed_table_v.py `
  --equations advection diffusion `
  --simulations 800

python .\run_sed_table_x.py `
  --benchmarks diffusion-convection joule1 `
  --simulations 300
```

### Ablation Experiments

```
python ".\ablation experiment\ab_compare_mcts_rollout.py" `
  --seeds 0 1 2 `
  --max_expressions 3000

python ".\ablation experiment\run_prior_bootstrap_ablation.py" `
  --seeds 0 1 2 `
  --max_expressions 1200
```
