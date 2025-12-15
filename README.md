# Elementary HPO

Elementary-hpo is a lightweight hyperparameter optimization library built on Sobol Sequences (Quasi-Monte Carlo methods).

It is designed to offer a mathematically superior alternative to Grid Search and Random Search by generating low-discrepancy sequences that cover the hyperparameter search space more evenly and efficiently. Unlike standard Random Search, `elementary-hpo` allows for sequential optimization, you can pause a search, analyze results, and generate new hyperparameter candidates that mathematically fill the "gaps" of previous runs, without redundancy.

Based on concepts discussed in the research paper [Hyperparameter Optimization in Machine Learning](https://arxiv.org/abs/2410.22854). This package allows you to optimize any scikit-learn compatible estimator class (e.g., `SVC`, `XGBClassifier`, `GradientBoostingRegressor`) hyperparameters efficiently by covering the search space more evenly than random search.

## Installation

```bash
pip install elementary-hpo
```

Using Poetry
If you are using Poetry for your project, add it as a dependency:
```bash
poetry add elementary-hpo
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

## Citation
If you use this package, please consider citing the foundational paper:

"Hyperparameter Optimization in Machine Learning" (2024). arXiv:2410.22854. Available at: https://arxiv.org/abs/2410.22854

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

#### **`LICENSE`**

```text
This project is licensed under the Apache License - see the LICENSE file for details.