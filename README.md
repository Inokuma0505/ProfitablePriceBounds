# Price Range Experiment Repository

This repository contains the experiment code and reference outputs for a synthetic multi-product pricing study.
The code compares several ways of constructing price ranges before solving a revenue maximization problem under a fitted linear demand model.

The repository is being prepared for public release alongside a paper. At this stage, the most important rule is:

- `paper_*.json` are reference artifacts and must not be changed unintentionally.

## What Is In This Repository

- `src/main_exp_test_par.py`
  Canonical experiment script for the `M=5` setting.
- `src/main_exp_test_par copy.py`
  Canonical experiment script for the `M=10` setting.
- `paper_1000.json`, `paper_300.json`, `paper_10_1000.json`, `paper_10_300.json`
  Reference results used for the paper.
- `docs/elasticity-experiments.md`
  Notes on the additional elasticity-based range construction methods.
- `plot_paper_results.py` and notebooks under `notebbok/`
  Figure-generation and exploratory notebooks. Notebooks are kept as supplementary materials, not as the main source of truth.
- `artifacts/intermediate/`, `artifacts/debug/`, `artifacts/figures/`
  Non-paper intermediate JSON files, debug outputs, and generated figures.

## Methods Included

The paper JSON files contain results for the following groups of methods.

- `so`: optimization with the true demand model.
- `po`: optimization with the fitted demand model on the full price range.
- `quan*`: quantile-based price ranges.
- `boot*`: bootstrap-based price ranges.
- `ebpa*`: cross-validation / penalty-based price ranges.
- `fixed*`: fixed-width price ranges centered at a reference price.
- `elaopt*`, `elabias*`: additional elasticity-based range construction methods.

## Environment Setup

The project currently uses `rye`.

```bash
rye sync
```

If you want linting and type checking:

```bash
make lint
```

## Reference Outputs

The following files are the primary public artifacts.

- `paper_1000.json`: `M=5`, `N=1000`
- `paper_300.json`: `M=5`, `N=300`
- `paper_10_1000.json`: `M=10`, `N=1000`
- `paper_10_300.json`: `M=10`, `N=300`

These files are the outputs that should be preserved across refactoring.
When checking reproducibility, numerical differences on the order of `1e-12` are acceptable, but larger changes should be treated as regressions.

## Reproducing Paper Results

The original experiment workflow uses the canonical scripts directly.

### `M=5`

```bash
rye run python src/main_exp_test_par.py
rye run python src/main_exp_test_par.py --n 300
```

### `M=10`

```bash
rye run python "src/main_exp_test_par copy.py"
rye run python "src/main_exp_test_par copy.py" --n 300
```

These commands follow the scripts' built-in output behavior.

## Current Repository Policy

- Preserve `paper_*.json` as the first-class reference outputs.
- Keep notebooks in the repository as supplementary materials.
- Avoid refactors that change experiment behavior before reproducibility checks are in place.
- Prefer documenting canonical scripts and outputs before restructuring code.
- Keep non-paper outputs under `artifacts/` when possible.

## Recommended Cleanup Direction

For a public paper repository, the practical recommendation is:

- Keep `paper_*.json` visible and easy to find.
- Keep non-paper intermediate outputs and generated figures under `artifacts/`.
- Delay large code moves until a stronger regression check for the paper JSON files is in place.

This keeps the repository understandable for readers while minimizing the risk of changing published results.
