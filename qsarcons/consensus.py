import random
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error,  roc_auc_score
from scipy.stats import spearmanr
from .genopt import Individual, GeneticAlgorithm

METRIC_MODES = {
    "mae": "minimize",
    "rmse": "minimize",
    "r2": "maximize",
    "rank": "maximize",
    "roc_auc_score": "maximize",
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
    elif metric == 'auto':
        if all(isinstance(v, (int, float)) for v in y_true):
            mae_norm = 1 / (1 + mean_absolute_error(y_true, y_pred))
            rmse_norm = 1 / (1 + root_mean_squared_error(y_true, y_pred))
            r2_norm = max(0.0, r2_score(y_true, y_pred))
            spearmanr_norm = max(0.0, spearmanr(y_true, y_pred)[0])
            return np.mean([mae_norm, rmse_norm, r2_norm, spearmanr_norm])
        else:
            roc_auc = roc_auc_score(y_true, y_pred)
            return roc_auc

class ConsensusSearch:
    """Base class for consensus model selection."""

    def __init__(self, cons_size=9, cons_size_candidates=None, metric=None):
        self.cons_size = cons_size
        self.cons_size_candidates = cons_size_candidates or [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.metric = metric
        self.n_filtered_models = None

    def _get_baseline_prediction(self, y: List) -> List:
        return list(np.mean(y) for _ in y)

    def _filter_models(self, x: DataFrame, y: List) -> DataFrame:
        """Filter out underperformed models based on baseline metric performance."""

        metric = "r2" if detect_task_type(y) == "regression" else "roc_auc_score"

        mode = METRIC_MODES[metric]
        baseline_pred = self._get_baseline_prediction(y)
        baseline_score = calc_accuracy(y, baseline_pred, metric=metric)

        filtered_cols = [col for col in x.columns if
                         (mode == 'maximize' and calc_accuracy(y, x[col], metric=metric) > baseline_score) or
                         (mode == 'minimize' and calc_accuracy(y, x[col], metric=metric) < baseline_score)]

        filtered = x[filtered_cols]
        self.n_filtered_models = filtered.shape[1]
        if self.n_filtered_models == 0:
            print("No models left after filtering. All models selected.")
            return x
        return filtered

    def run(self, x: DataFrame, y: List) -> List:
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

    def __init__(self, cons_size=10, n_iter=5000, metric="mae", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
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

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) :
        """Run systematic search for regression models."""
        scores = [(col, calc_accuracy(y, x[col], self.metric)) for col in x.columns]
        scores.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        top_cols = [col for col, _ in scores[:cons_size]]
        return list(top_cols)

class GeneticSearch(ConsensusSearch):
    """Genetic algorithm-based search for optimal regression consensus. """
    def __init__(self, cons_size=10, n_iter=200, pop_size=50, mut_prob=0.2, metric="mae", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.mut_prob = mut_prob

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        """Run genetic algorithm search for a fixed consensus size."""

        def objective(ind: Individual) -> float:
            y_pred = self.predict_cons(x.iloc[:, list(ind)])
            return calc_accuracy(y, y_pred, self.metric)

        space = range(len(x.columns))
        task = METRIC_MODES[self.metric]
        ga = GeneticAlgorithm(task=task, pop_size=self.pop_size, crossover_prob=0.90,
                              mutation_prob=self.mut_prob, elitism=True, random_seed=11)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=cons_size)
        ga.run(n_iter=self.n_iter)
        return x.columns[list(ga.get_global_best())]
