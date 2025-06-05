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
    "rank": "maximize",
    "top": "maximize",
}

def top_x_overlap_rate(y_true, y_pred, top_percent=0.1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_top = int(len(y_true) * top_percent)
    if n_top < 1:
        raise ValueError("Top percent too small for dataset size.")

    top_true_idx = set(np.argsort(y_true)[-n_top:])  # highest experimental
    top_pred_idx = set(np.argsort(y_pred)[-n_top:])  # highest predicted

    overlap = len(top_true_idx & top_pred_idx)
    return overlap / n_top


def calc_accuracy(y_true, y_pred, metric='mae'):
    y_true, y_pred = list(y_true), list(y_pred)

    if metric == 'mae':
        acc = mean_absolute_error(y_true, y_pred)
    elif metric == 'r2':
        acc = r2_score(y_true, y_pred)
    elif metric == 'rmse':
        acc = root_mean_squared_error(y_true, y_pred)
    elif metric == 'rank':
        acc, p_value = spearmanr(y_true, y_pred)
        try:
            acc = acc.item()
        except:
            pass
    elif metric == 'top':
        acc = top_x_overlap_rate(y_true, y_pred)

    return acc


class SearchRegressor:
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

        self.n_filtered_models = None  # number of models after filtering

    def _filter_models(self, x: DataFrame, y: Series) -> DataFrame:
        mode = METRIC_MODES[self.metric]

        if self.metric == "rank":
            baseline_score = 0.0  # no correlation baseline
        else:
            baseline_pred = np.full_like(y, fill_value=y.mean(), dtype=np.float64)
            baseline_score = calc_accuracy(y, baseline_pred, self.metric)

        filtered_cols = []
        for col in x.columns:
            score = calc_accuracy(y, x[col], self.metric)
            if mode == "maximize" and score > baseline_score:
                filtered_cols.append(col)
            elif mode == "minimize" and score < baseline_score:
                filtered_cols.append(col)

        filtered_df = x[filtered_cols]
        self.n_filtered_models = filtered_df.shape[1]

        if self.n_filtered_models == 0:
            raise ValueError(
                f"No models left after filtering worse than baseline (score={baseline_score:.4f})"
            )
        return filtered_df

    def run(self, x: DataFrame, y: Series) -> Index:
        x_filtered = self._filter_models(x, y)

        if isinstance(self.cons_size, int):
            return self._run_with_cons_size(x_filtered, y, self.cons_size)

        elif self.cons_size == 'auto':
            best_cons = None
            best_score = None
            mode = METRIC_MODES[self.metric]

            for size in self.cons_size_candidates:
                candidate_cons = self._run_with_cons_size(x_filtered, y, size)
                y_cons = x_filtered[candidate_cons].mean(axis=1)
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
            crossover_prob=0.95,
            mutation_prob=self.mut_prob,
            elitism=True
        )
        ga.set_fitness(objective)

        initial_best = SystematicSearchRegressor(cons_size=cons_size, metric=self.metric).run(x, y)
        elite_indices = [x.columns.get_loc(col) for col in initial_best]
        elite_individual = Individual(elite_indices)

        ga.initialize(space, ind_size=cons_size, ind_elite=elite_individual)
        ga.run(n_iter=self.n_iter, verbose=False)

        best_cons = list(ga.get_global_best())
        return x.columns[best_cons]

