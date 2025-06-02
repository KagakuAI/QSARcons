import random
from typing import List, Union, Callable, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr

from .genopt import Individual, GeneticAlgorithm

# Mapping of metrics to objective mode
METRIC_MODES = {
    "mae": "minimize",
    "rmse": "minimize",
    "r2": "maximize",
    "spearmanr": "maximize"
}


def calc_accuracy(y_true: Union[Series, List[float]],
                  y_pred: Union[Series, List[float]],
                  metric: str = 'mae') -> float:
    """
    Calculate accuracy metric between predicted and true values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        metric: Metric to use ('mae', 'r2', 'rmse', 'spearmanr').

    Returns:
        Calculated metric score.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'r2':
        return r2_score(y_true, y_pred)
    elif metric == 'rmse':
        return root_mean_squared_error(y_true, y_pred)
    elif metric == 'spearmanr':
        return spearmanr(y_true, y_pred).correlation
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class SearchRegressor:
    """
    Base class for search-based regressors supporting optional consensus size optimization.

    Args:
        cons_size: int for fixed size, or 'auto' for automatic selection.
        cons_size_candidates: list of consensus sizes to try if cons_size='auto'.
        metric: performance metric to optimize.
    """

    def __init__(self,
                 cons_size: Union[int, str] = 10,
                 cons_size_candidates: Optional[List[int]] = None,
                 metric: str = "mae") -> None:
        self.cons_size = cons_size
        self.metric = metric
        if cons_size_candidates is None:
            self.cons_size_candidates = [3, 5, 7, 9, 11, 13, 15]
        else:
            self.cons_size_candidates = cons_size_candidates

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        """
        Override this method in subclasses to run search with a fixed cons_size.
        """
        raise NotImplementedError

    def run(self, x: DataFrame, y: Series) -> Index:
        """
        Run search with either fixed consensus size or automatic consensus size optimization.
        """
        if isinstance(self.cons_size, int):
            # Fixed cons_size
            return self._run_with_cons_size(x, y, self.cons_size)

        elif self.cons_size == 'auto':
            best_cons = None
            best_score = None
            mode = METRIC_MODES[self.metric]

            for size in self.cons_size_candidates:
                candidate_cons = self._run_with_cons_size(x, y, size)
                y_cons = x[candidate_cons].mean(axis=1)
                score = calc_accuracy(y, y_cons, self.metric)

                if best_score is None:
                    best_score = score
                    best_cons = candidate_cons
                else:
                    if mode == "minimize" and score < best_score:
                        best_score = score
                        best_cons = candidate_cons
                    elif mode == "maximize" and score > best_score:
                        best_score = score
                        best_cons = candidate_cons

            return best_cons

        else:
            raise ValueError(f"Unsupported cons_size value: {self.cons_size}")


class RandomSearchRegressor(SearchRegressor):
    def __init__(self, cons_size=10, n_iter=5000, metric="mae", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.n_iter = n_iter

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        cons_list = []

        for _ in range(self.n_iter):
            random_cons = random.sample(range(len(x.columns)), k=cons_size)
            y_cons = x.iloc[:, random_cons].mean(axis=1)
            acc = calc_accuracy(y, y_cons, metric=self.metric)
            cons_list.append((random_cons, acc))

        reverse = METRIC_MODES[self.metric] == "maximize"
        cons_list.sort(key=lambda item: item[1], reverse=reverse)

        best_cons = cons_list[0][0]
        return x.columns[list(best_cons)]


class SystematicSearchRegressor(SearchRegressor):
    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        scores = [(col, calc_accuracy(y, x[col], self.metric)) for col in x.columns]
        reverse = METRIC_MODES[self.metric] == "maximize"
        sorted_scores = sorted(scores, key=lambda item: item[1], reverse=reverse)

        best_cons = [col for col, _ in sorted_scores[:cons_size]]
        return x[best_cons].columns


class GeneticSearchRegressor(SearchRegressor):
    def __init__(self, cons_size=10, n_iter=200, pop_size=50, mut_prob=0.2, metric="mae", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.mut_prob = mut_prob

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        def objective(cons: Individual) -> float:
            y_cons = x.iloc[:, list(cons)].mean(axis=1)
            return calc_accuracy(y, y_cons, metric=self.metric)

        space = range(len(x.columns))
        task = METRIC_MODES[self.metric]

        ga = GeneticAlgorithm(
            task=task,
            pop_size=self.pop_size,
            crossover_prob=0.8,
            mutation_prob=self.mut_prob,
            elitism=True
        )
        ga.set_fitness(objective)

        initial_best = SystematicSearchRegressor(cons_size=cons_size, metric=self.metric).run(x, y)
        elite_indices = [x.columns.get_loc(col) for col in initial_best]
        elite_individual = Individual(elite_indices)

        ga.initialize(space, ind_size=cons_size, ind_elite=elite_individual)
        ga.run(n_iter=self.n_iter, verbose=False)

        best_cons = list(ga.best_individual())
        return x.columns[best_cons]

