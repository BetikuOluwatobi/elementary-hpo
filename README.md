# Elementary HPO

A lightweight hyperparameter optimization tool using Quasi-Monte Carlo (Sobol) sequences. This package allows you to optimize any scikit-learn compatible estimator class (e.g., `SVC`, `XGBClassifier`, `GradientBoostingRegressor`) hyperparameters efficiently by covering the search space more evenly than random search.

## Installation

```bash
pip install elementary-hpo
```

## Quick Start

### 1. Basic Usage (Random Forest)
```python
from sklearn.datasets import make_classification
from elementary_hpo import SobolOptimizer, plot_optimization_results, plot_space_coverage

# 1. Generate Data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. Define Search Space
param_bounds = {
    'n_estimators': (50, 300),          # Integer tuple = Numerical range
    'max_depth': (3, 20),
    'min_samples_split': (0.01, 0.5),   # Float tuple = Numerical range
    'criterion': ['gini', 'entropy']    # List = Categorical choices
}

# 3. Initialize Optimizer
optimizer = SobolOptimizer(param_bounds)

# 4. Run Optimization (Phase 1)
optimizer.optimize(X, y, n_samples=8, batch_name="Batch 1")

# 5. Extend Optimization (Phase 2 - fills gaps in Phase 1)
optimizer.optimize(X, y, n_samples=8, batch_name="Batch 2")

# 6. Get Results
print(optimizer.get_best_params())
plot_optimization_results(optimizer.results)
plot_space_coverage(optimizer.results, x_col="n_estimators", y_col="max_depth")
```

#### **`LICENSE`** (MIT)

```text
MIT License

Copyright (c) 2025 Oluwatobi Betiku

Permission is hereby granted, free of charge, to any person obtaining a copy...
[Standard MIT Text]