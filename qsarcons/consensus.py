import random
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from scipy.stats import spearmanr
from .genopt import Individual, GeneticAlgorithm

METRIC_MODES = {
    "mae": "minimize",
    "rmse": "minimize",
    "r2": "maximize",
    "rank": "maximize",
    "balanced_accuracy_score": "maximize",
    "auto": "maximize",
}

def detect_task_type(y):
    y = pd.Series(y).dropna()
    return "classification" if y.nunique() == 2 else "regression"

def calc_accuracy(y_true, y_pred, metric=None):
    """Compute performance metrics for regression or classification tasks."""

    y_true, y_pred = list(y_true), list(y_pred)

    if metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'rmse':
        return root_mean_squared_error(y_true, y_pred)
    elif metric == 'r2':
        return r2_score(y_true, y_pred)
    elif metric == 'rank':
        acc, _ = spearmanr(y_true, y_pred)
        return acc.item() if hasattr(acc, 'item') else acc
    elif metric == 'balanced_accuracy_score':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == 'auto':
        if detect_task_type(y_true) == "regression":
            mae_norm = 1 / (1 + mean_absolute_error(y_true, y_pred))
            rmse_norm = 1 / (1 + root_mean_squared_error(y_true, y_pred))
            r2_norm = max(0.0, r2_score(y_true, y_pred))
            spearmanr_norm = max(0.0, spearmanr(y_true, y_pred)[0])
            return np.mean([mae_norm, rmse_norm, r2_norm, spearmanr_norm])
        elif detect_task_type(y_true) == "classification":
            return balanced_accuracy_score(y_true, y_pred)

class ConsensusSearch:
    """Base class for consensus model selection."""

    def __init__(self, cons_size=9, cons_size_candidates=None, metric=None):
        self.cons_size = cons_size
        self.cons_size_candidates = cons_size_candidates or range(2, 16, 2)
        self.metric = metric
        self._task_type = None

    def _filter_models(self, x: DataFrame, y: List) -> DataFrame:
        """Filter out underperformed models based on baseline metric performance."""

        if self._task_type == "classification":
            x = (x >= 0.5).astype(int)

        metric = "r2" if self._task_type == "regression" else "balanced_accuracy_score"
        mode = METRIC_MODES[metric]
        baseline_score = 0 if self._task_type == "regression" else 0.5

        filtered_cols = [col for col in x.columns if
                         (mode == 'maximize' and calc_accuracy(y, x[col], metric=metric) > baseline_score) or
                         (mode == 'minimize' and calc_accuracy(y, x[col], metric=metric) < baseline_score)]

        filtered = x[filtered_cols]
        if filtered.shape[1] == 0:
            print("No models left after filtering. All models selected.")
            return x
        return filtered

    def _run_with_cons_size(self, x, y, cons_size):
        if self._task_type == "classification":
            x = (x >= 0.5).astype(int)
        return self._run_with_cons_size(x, y, cons_size)

    def run(self, x: DataFrame, y: List):
        """Execute consensus model search."""

        self._task_type = detect_task_type(y)

        x_filtered = self._filter_models(x, y)
        if len(x_filtered.columns) < max(self.cons_size_candidates):
            print("WARNING: The number of filtered models is lower than the consensus size candidates. All models are used for consensus search.")
            x_filtered = x

        if isinstance(self.cons_size, int):
            return self._run_with_cons_size(x_filtered, y, self.cons_size)

        elif self.cons_size == 'auto':
            best_cons = None
            best_score = None
            mode = METRIC_MODES[self.metric]
            for size in self.cons_size_candidates:
                candidate = self._run_with_cons_size(x_filtered, y, size)
                y_pred = self.predict(x_filtered[candidate])
                score = calc_accuracy(y, y_pred, self.metric)
                if best_score is None or \
                   (mode == 'maximize' and score > best_score) or \
                   (mode == 'minimize' and score < best_score):
                    best_score = score
                    best_cons = candidate
            return list(best_cons)

    def predict(self, x_subset: DataFrame) -> List:

        if self._task_type == "regression":
            return list(x_subset.mean(axis=1))

        elif self._task_type == "classification":
            # Convert probabilities to 0/1 per model
            binary_preds = (x_subset >= 0.5).astype(int)

            # Majority voting across models
            votes = binary_preds.sum(axis=1)
            majority = (votes >= (binary_preds.shape[1] / 2)).astype(int)
            return list(majority)
        else:
            raise ValueError("Task type not set. Run `.run()` first.")

    def predict_proba(self, x_subset: DataFrame) -> List:

        if self._task_type == "regression":
            raise ValueError("predict_proba is not available for regression consensus models.")
        elif self._task_type == "classification":
            return list(x_subset.mean(axis=1))
        else:
            raise ValueError("Task type not set. Run `.run()` first.")

class RandomSearch(ConsensusSearch):
    """Randomized search for optimal regression consensus."""

    def __init__(self, n_iter=1000, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        """Run random search for a fixed consensus size."""
        results = []
        for _ in range(self.n_iter):
            cols = random.sample(list(x.columns), cons_size)
            y_pred = self.predict(x[cols])
            score = calc_accuracy(y, y_pred, self.metric)
            results.append((cols, score))
        results.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        return pd.Index(results[0][0])

class SystematicSearch(ConsensusSearch):
    """Systematic selection of top-performing regression models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int):
        """Run systematic search for regression models."""

        scores = [(col, calc_accuracy(y, x[col], self.metric)) for col in x.columns]
        scores.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        top_cols = [col for col, _ in scores[:cons_size]]
        return list(top_cols)

class GeneticSearch(ConsensusSearch):
    """Genetic algorithm-based search for optimal regression consensus. """

    def __init__(self, n_iter=50, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter
        self.verbose = verbose

    def _run_with_cons_size(self, x, y, cons_size) -> Index:

        def objective(ind: Individual) -> float:
            y_pred = self.predict(x.iloc[:, list(ind)])
            return calc_accuracy(y, y_pred, self.metric)

        space = range(len(x.columns))
        task = METRIC_MODES[self.metric]
        ga = GeneticAlgorithm(task=task, pop_size=100, crossover_prob=0.90, mutation_prob=0.2, elitism=True, verbose=False)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=cons_size)
        ga.run(n_iter=self.n_iter)

        return x.columns[list(ga.get_global_best())]
