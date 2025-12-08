import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
from sklearn.base import is_classifier


DEFAULT_PARAM_GRID_REGRESSORS = {
    "Ridge": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "solver": ["auto", "saga", "lsqr"],
    },
    "PLSRegression": {
        "n_components": [2, 4, 8, 16, 32],
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
    "CatBoostRegressor": {
        "iterations": [100, 300, 500],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
        "bagging_temperature": [0, 0.5, 1.0],
        "border_count": [32, 64, 128],
    },
    "MLPRegressor": {
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-4, 1e-3],
        "hidden_layer_sizes": [(128,), (512, 256, 128), (2048, 1024, 512, 256, 128, 64)],
        "max_iter": [300, 1000],
    },
    "SVR": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "LinearSVR": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.001, 0.01, 0.1, 0.5],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
        "max_iter": [1000, 5000, 10000],
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
        "max_iter": [500, 2000],
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
    "CatBoostClassifier": {
        "iterations": [100, 300, 500],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
        "bagging_temperature": [0, 0.5, 1.0],
        "border_count": [32, 64, 128],
    },
    "MLPClassifier": {
        "activation": ["relu", "tanh"],
        "learning_rate_init": [1e-4, 1e-3],
        "hidden_layer_sizes": [(128,), (512, 256, 128), (2048, 1024, 512, 256, 128, 64)],
        "max_iter": [300, 1000],
         },
    "SVC": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"],
    },
    "LinearSVC": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "loss": ["hinge", "squared_hinge"],
        "penalty": ["l2"],
        "max_iter": [1000, 5000, 10000],
    },
}

def get_predictions(estimator, X):
    if is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1].tolist()
    else:
        return estimator.predict(X).tolist()

def single_split_score(est, x, y, scoring, test_size=0.4, random_state=42):
    """
    Performs a single train/val split inside the function,
    fits the estimator, and returns the validation score.
    """

    x_train, x_val, y_train, y_val = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
    )

    est.fit(x_train, y_train)
    y_pred = get_predictions(est, x_val)

    scorer = get_scorer(scoring)
    score = scorer._score_func(y_val, y_pred)

    return score


class StepwiseHopt:
    """
    Stepwise hyperparameter optimization for scikit-learn estimators.

    This optimizer iteratively tunes each hyperparameter one at a time
    while keeping the other parameters fixed at their current best values.
    """

    def __init__(self, estimator, param_grid, scoring=None, verbose=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.verbose = verbose
        self.best_params_ = {}

    def _evaluate_model(self, param, val, x, y, best_params, n_jobs):

        params = {**best_params, param: val}
        est = self.estimator.__class__(**params)
        score = single_split_score(est, x, y, scoring=self.scoring)
        return val, score

    def fit(self, x, y):

        if self.verbose:
            total_steps = sum(len(v) for v in self.param_grid.values())
            print(f"Stepwise optimization started with {total_steps} options")

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
            args = [(param, val, x, y, best_params, n_jobs) for val in options]
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
        self.estimator = self.estimator.__class__(**best_params)
        total_time_min = (time.time() - start_time) / 60
        if self.verbose:
            print(f"\nStepwise optimization completed in {total_time_min:.1f} min")

        return self
