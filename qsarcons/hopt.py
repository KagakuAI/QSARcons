import time
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from joblib import parallel_backend
import numpy as np

DEFAULT_PARAM_GRID_REGRESSORS = {
    "Ridge": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "solver": ["auto", "saga", "lsqr"],
    },
    "PLSRegression": {
        "n_components": [2, 4, 8, 16, 32],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [1, 3, 5, 9, 15, 25],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "DecisionTreeRegressor": {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10, 20],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
    },
    "MLPRegressor": {
        "hidden_layer_sizes": [(100,), (200, 100), (200, 100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
    },
    "SVR": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
}

DEFAULT_PARAM_GRID_CLASSIFIERS = {
    "RidgeClassifier": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "solver": ["auto", "saga", "lsqr"],
    },
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["liblinear", "lbfgs", "saga"],
        "penalty": ["l2", "l1"],
        "max_iter": [500, 2000],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [1, 3, 5, 9, 15, 25],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10, 20],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [5, 10, 20, None],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [(100,), (200, 100), (200, 100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 1e-2],
        "max_iter": [200, 500],
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
}

def get_optimal_threads(n_jobs: int) -> int:
    total_cpus = os.cpu_count() or 1
    return max(1, total_cpus // n_jobs)

class StepwiseHopt:
    """Stepwise hyperparameter optimizer for sklearn models."""

    def __init__(self, estimator, param_grid, scoring=None, cv=3, verbose=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.best_params_ = {}

    def _evaluate_model(self, param, val, X, y, best_params, n_jobs):
        threads = get_optimal_threads(n_jobs)
        est = clone(self.estimator)
        est.set_params(**{**best_params, param: val})

        # Evaluate via CV
        with parallel_backend("threading", n_jobs=threads):
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
        mean_score = np.mean(scores)
        return val, mean_score

    def fit(self, X, y):
        total_steps = sum(len(v) for v in self.param_grid.values())
        current_step = 0
        start_time = time.time()

        best_params = {}
        for param, options in self.param_grid.items():
            if not isinstance(options, (list, tuple)):
                best_params[param] = options
                continue

            if self.verbose:
                print(f"\nOptimizing '{param}' ({len(options)} options)")

            n_jobs = len(options)
            args = [(param, val, X, y, best_params, n_jobs) for val in options]
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(lambda a: self._evaluate_model(*a), args))

            # Select best value
            if self.scoring is None or "neg" in str(self.scoring):
                best_val, best_score = max(results, key=lambda x: x[1])  # higher is better
            else:
                best_val, best_score = max(results, key=lambda x: x[1])

            best_params[param] = best_val
            current_step += len(options)
            if self.verbose:
                print(f"→ Best {param}: {best_val}, score={best_score:.4f}")

        self.best_params_ = best_params
        self.estimator.set_params(**best_params)
        total_time_min = (time.time() - start_time) / 60
        if self.verbose:
            print(f"\nStepwise optimization completed in {total_time_min:.1f} min")

        return self
