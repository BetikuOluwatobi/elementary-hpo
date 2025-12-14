import pytest
import numpy as np
from sklearn.svm import SVC
from elementary_hpo.utils import map_categorical
from elementary_hpo.core import SobolOptimizer

def test_map_categorical():
    params = {'criterion': ['a', 'b']}
    n_iter = 4
    result = map_categorical(n_iter, params)
    
    assert 'criterion' in result
    assert len(result['criterion']) == 4
    # Check if distribution is roughly even (2 of each for 4 samples)
    unique, counts = np.unique(result['criterion'], return_counts=True)
    assert len(unique) == 2

def test_optimizer_generalization():
    """Test using a different model (SVM) instead of Random Forest."""
    # Dummy data
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    # Define parameter bounds
    bounds = {
        'C': (0.1, 10.0),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': (0.001, 1.0)
    }
    
    # Initialize with SVC class
    opt = SobolOptimizer(bounds, model_class=SVC, fixed_params={'probability': True}, random_state=1)
    
    # Run small batch
    df = opt.optimize(X, y, n_samples=6, batch_name="SVM Test Batch", metric='accuracy', cv=3, n_jobs=1)
    
    assert not df.empty
    assert 'score' in df.columns
    assert 'C' in df.columns
    assert 'kernel' in df.columns
    assert len(df) == 6

def test_sobol_power_of_two_warning():
    """Test that a warning is raised when n_samples is not a power of two."""
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)

    bounds = {
        'param1': (0, 1),
        'param2': (0, 10)
    }
    
    opt = SobolOptimizer(bounds, model_class=SVC, random_state=1)
    
    # 3 is not a power of 2, should warn
    with pytest.warns(UserWarning, match="not a power of 2"):
        opt.optimize(X, y, n_samples=3, batch_name="Power of Two Test", metric='accuracy', cv=3, n_jobs=1)
