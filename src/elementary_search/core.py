import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Optional, Tuple
from .utils import map_parameters

class SobolOptimizer:
    def __init__(self, param_bounds: Dict[str, Any], random_state: int = 42):
        """
        Initialize the optimizer with parameter bounds.
        
        Args:
            param_bounds: Dictionary defining bounds (tuples for numerical, lists for categorical).
            random_state: Seed for reproducibility (though Sobol is deterministic, shuffling isn't).
        """
        self.param_bounds = param_bounds
        self.numerical_keys = [k for k, v in param_bounds.items() if isinstance(v, tuple)]
        self.d = len(self.numerical_keys)
        
        # Initialize Sobol Engine
        # Scramble=True allows expanding the sequence later
        self.sampler = qmc.Sobol(d=self.d, scramble=True, seed=random_state)
        
        self.results = pd.DataFrame()

    def _evaluate_batch(self, 
                        params_list: list, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        batch_name: str, 
                        metric: str = 'accuracy', 
                        cv: int = 3) -> pd.DataFrame:
        
        results_data = []
        print(f"--- Processing {batch_name} ---")

        for param in params_list:
            # Model instantiation (Hardcoded to RF based on notebook, could be generalized)
            rf = RandomForestClassifier(**param, random_state=42)
            
            # Cross Validation
            scores = cross_val_score(rf, X, y, cv=cv, scoring=metric)
            mean_score = scores.mean()

            entry = param.copy()
            entry['score'] = mean_score
            entry['batch'] = batch_name
            results_data.append(entry)

        return pd.DataFrame(results_data)

    def optimize(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 n_samples: int, 
                 batch_name: str = "Batch",
                 metric: str = 'accuracy') -> pd.DataFrame:
        """
        Generates 'n_samples' new configurations and evaluates them.
        """
        # Sobol works best with powers of 2, warn if not? (optional optimization)
        
        # Generate raw samples from unit hypercube
        sample_batch = self.sampler.random(n=n_samples)
        
        # Map to actual parameters
        params = map_parameters(sample_batch, self.param_bounds)
        
        # Evaluate
        new_results = self._evaluate_batch(params, X, y, batch_name, metric)
        
        # Append to history
        self.results = pd.concat([self.results, new_results], ignore_index=True)
        
        return new_results

    def get_best_params(self) -> Dict[str, Any]:
        if self.results.empty:
            return {}
        best_row = self.results.loc[self.results['score'].idxmax()]
        # Filter out metadata columns
        return {k: v for k, v in best_row.items() if k not in ['score', 'batch']}