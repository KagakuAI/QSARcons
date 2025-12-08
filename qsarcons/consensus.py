import random
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from scipy.stats import spearmanr
from .genopt import Individual, GeneticAlgorithm

METRIC_MODES = {
    "mae": "minimize",
    "rmse": "minimize",
    "r2": "maximize",
    "rank": "maximize",
    "roc_auc_score": "maximize",
    "average_precision_score": "maximize",
    "log_loss": "maximize",
    "brier_score_loss": "maximize",
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
    elif metric == 'roc_auc_score':
        return roc_auc_score(y_true, y_pred)
    elif metric == 'average_precision_score':
        return average_precision_score(y_true, y_pred)
    elif metric == 'auto':
        if all(isinstance(v, (int, float)) for v in y_true):
            mae_norm = 1 / (1 + mean_absolute_error(y_true, y_pred))
            rmse_norm = 1 / (1 + root_mean_squared_error(y_true, y_pred))
            r2_norm = max(0.0, r2_score(y_true, y_pred))
            spearmanr_norm = max(0.0, spearmanr(y_true, y_pred)[0])
            return np.mean([mae_norm, rmse_norm, r2_norm, spearmanr_norm])
        else:
            return np.mean([roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)])

class ConsensusSearch:
    """Base class for consensus model selection."""

    def __init__(self, cons_size=9, cons_size_candidates=None, metric=None):
        self.cons_size = cons_size
        self.cons_size_candidates = cons_size_candidates or [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.metric = metric

    def _filter_models(self, x: DataFrame, y: List) -> DataFrame:
        """Filter out underperformed models based on baseline metric performance."""

        metric = "r2" if detect_task_type(y) == "regression" else "roc_auc_score"
        mode = METRIC_MODES[metric]
        baseline_score = 0 if detect_task_type(y) == "regression" else 0.5

        filtered_cols = [col for col in x.columns if
                         (mode == 'maximize' and calc_accuracy(y, x[col], metric=metric) > baseline_score) or
                         (mode == 'minimize' and calc_accuracy(y, x[col], metric=metric) < baseline_score)]

        filtered = x[filtered_cols]
        if filtered.shape[1] == 0:
            print("No models left after filtering. All models selected.")
            return x
        return filtered

    def _run_with_cons_size(self, x, y, cons_size):
        return NotImplementedError

    def run(self, x: DataFrame, y: List):
        """Execute consensus model search."""

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
                y_pred = self.predict_cons(x_filtered[candidate])
                score = calc_accuracy(y, y_pred, self.metric)
                if best_score is None or \
                   (mode == 'maximize' and score > best_score) or \
                   (mode == 'minimize' and score < best_score):
                    best_score = score
                    best_cons = candidate
            return list(best_cons)

    def predict_cons(self, x_subset: DataFrame) -> List:
        return list(x_subset.mean(axis=1))

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
            y_pred = self.predict_cons(x[cols])
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

    def __init__(self, n_iter=50, **kwargs):
        super().__init__(**kwargs)
        self.n_iter = n_iter

    def _run_with_cons_size(self, x, y, cons_size) -> Index:

        def objective(ind: Individual) -> float:
            y_pred = self.predict_cons(x.iloc[:, list(ind)])
            return calc_accuracy(y, y_pred, self.metric)

        space = range(len(x.columns))
        task = METRIC_MODES[self.metric]
        ga = GeneticAlgorithm(task=task, pop_size=50, crossover_prob=0.90, mutation_prob=0.2, elitism=True)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=cons_size)
        ga.run(n_iter=self.n_iter)

        return x.columns[list(ga.get_global_best())]
