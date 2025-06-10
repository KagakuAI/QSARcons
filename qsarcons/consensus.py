import random
from typing import List, Union, Optional, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, Index

from sklearn.metrics import (
    mean_absolute_error, r2_score, root_mean_squared_error,
    accuracy_score, f1_score
)
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score

from scipy.stats import spearmanr

from .genopt import Individual, GeneticAlgorithm


METRIC_MODES = {
    "mae": "minimize",
    "rmse": "minimize",
    "r2": "maximize",
    "rank": "maximize",
    "top": "maximize",
    "acc": "maximize",
    "f1": "maximize",
    "auto": "maximize",  # <- Add this
}

def top_x_overlap_rate(y_true, y_pred, top_percent=0.1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_top = int(len(y_true) * top_percent)
    if n_top < 1:
        raise ValueError("Top percent too small for dataset size.")
    top_true_idx = set(np.argsort(y_true)[-n_top:])
    top_pred_idx = set(np.argsort(y_pred)[-n_top:])
    return len(top_true_idx & top_pred_idx) / n_top


def calc_accuracy(y_true, y_pred, metric='mae'):
    y_true, y_pred = list(y_true), list(y_pred)
    
    if metric == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric == 'r2':
        return r2_score(y_true, y_pred)
    elif metric == 'rmse':
        return root_mean_squared_error(y_true, y_pred)
    elif metric == 'rank':
        acc, _ = spearmanr(y_true, y_pred)
        return acc.item() if hasattr(acc, 'item') else acc
    elif metric == 'top':
        return top_x_overlap_rate(y_true, y_pred)
    elif metric == 'acc':
        return accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    elif metric == 'auto':
        if all(isinstance(v, (int, float)) for v in y_true):
            # Regression combined metric
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            r2_val = r2_score(y_true, y_pred)
            if np.all(np.array(y_pred) == y_pred[0]) or np.all(np.array(y_true) == y_true[0]):
                rank = 0.0
            else:
                rank, _ = spearmanr(y_true, y_pred)
                rank = rank.item() if hasattr(rank, 'item') else rank
            mae_score = 1 / (1 + mae)
            rmse_score = 1 / (1 + rmse)
            r2_score_norm = max(0.0, r2_val)
            rank_score = max(0.0, rank)
            return np.mean([mae_score, rmse_score, r2_score_norm, rank_score])
        else:
            # Classification combined metric
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            return np.mean([acc, bal_acc, f1, precision, recall])


class ConsensusSearch:
    def __init__(self, cons_size: Union[int, str] = 10, cons_size_candidates: Optional[List[int]] = None, metric: str = "mae"):
        self.cons_size = cons_size
        self.metric = metric
        self.cons_size_candidates = cons_size_candidates or [3, 5, 7, 9, 11, 13, 15]
        self.n_filtered_models = None

    def _filter_models(self, x: DataFrame, y: Series) -> DataFrame:
        raise NotImplementedError

    def _consensus_predict(self, x_subset: DataFrame) -> Series:
        raise NotImplementedError

    def run(self, x: DataFrame, y: Series) -> Index:
        x_filtered = self._filter_models(x, y)
        if isinstance(self.cons_size, int):
            return self._run_with_cons_size(x_filtered, y, self.cons_size)
        elif self.cons_size == 'auto':
            best_cons = None
            best_score = None
            mode = METRIC_MODES[self.metric]
            for size in self.cons_size_candidates:
                candidate = self._run_with_cons_size(x_filtered, y, size)
                y_pred = self._consensus_predict(x_filtered[candidate])
                score = calc_accuracy(y, y_pred, self.metric)
                if best_score is None or \
                   (mode == 'maximize' and score > best_score) or \
                   (mode == 'minimize' and score < best_score):
                    best_score = score
                    best_cons = candidate
            return best_cons
        else:
            raise ValueError(f"Unsupported cons_size value: {self.cons_size}")


class ConsensusSearchRegressor(ConsensusSearch):
    def _filter_models(self, x: DataFrame, y: Series, metric: str = "r2") -> DataFrame:

        mode = METRIC_MODES[metric]
        
        baseline_pred = np.full_like(y, fill_value=y.mean(), dtype=np.float64)
        baseline_score = calc_accuracy(y, baseline_pred, metric=metric)
        
        filtered_cols = [col for col in x.columns if
                         (mode == 'maximize' and calc_accuracy(y, x[col], metric=metric) > baseline_score) or
                         (mode == 'minimize' and calc_accuracy(y, x[col], metric=metric) < baseline_score)]
        
        filtered = x[filtered_cols]
        self.n_filtered_models = filtered.shape[1]
        
        if self.n_filtered_models == 0:
            raise ValueError("No models left after filtering.")
        return filtered

    def _consensus_predict(self, x_subset: DataFrame) -> Series:
        return x_subset.mean(axis=1)


class ConsensusSearchClassifier(ConsensusSearch):
    def _filter_models(self, x: DataFrame, y: Series) -> DataFrame:
        mode = METRIC_MODES[self.metric]
        baseline_pred = pd.Series([y.mode()[0]] * len(y), index=y.index)
        baseline_score = calc_accuracy(y, baseline_pred, self.metric)
        filtered_cols = [col for col in x.columns if
                         (mode == 'maximize' and calc_accuracy(y, x[col], self.metric) > baseline_score) or
                         (mode == 'minimize' and calc_accuracy(y, x[col], self.metric) < baseline_score)]
        filtered = x[filtered_cols]
        self.n_filtered_models = filtered.shape[1]
        if self.n_filtered_models == 0:
            raise ValueError("No models left after filtering.")
        return filtered

    def _majority_vote(self, preds: pd.DataFrame) -> pd.Series:
        arr = preds.to_numpy()
        classes = np.unique(arr)
        counts = (arr[:, :, None] == classes).sum(axis=1)
        max_counts = counts.max(axis=1, keepdims=True)
        mask = counts == max_counts
        indices = np.argmax(mask, axis=1)
        modes = classes[indices]
        return pd.Series(modes, index=preds.index)

    def _consensus_predict(self, x_subset: DataFrame) -> Series:
        return self._majority_vote(x_subset)


class RandomSearchRegressor(ConsensusSearchRegressor):
    def __init__(self, cons_size=10, n_iter=5000, metric="mae", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.n_iter = n_iter

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        results = []
        for _ in range(self.n_iter):
            cols = random.sample(list(x.columns), cons_size)
            y_pred = self._consensus_predict(x[cols])
            score = calc_accuracy(y, y_pred, self.metric)
            results.append((cols, score))
        results.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        return pd.Index(results[0][0])


class RandomSearchClassifier(ConsensusSearchClassifier):
    def __init__(self, cons_size=10, n_iter=1000, metric="acc", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.n_iter = n_iter

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        results = []
        for _ in range(self.n_iter):
            cols = random.sample(list(x.columns), cons_size)
            y_pred = self._consensus_predict(x[cols])
            score = calc_accuracy(y, y_pred, self.metric)
            results.append((cols, score))
        results.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        return pd.Index(results[0][0])


class SystematicSearchRegressor(ConsensusSearchRegressor):
    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        scores = [(col, calc_accuracy(y, x[col], self.metric)) for col in x.columns]
        scores.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        top_cols = [col for col, _ in scores[:cons_size]]
        return pd.Index(top_cols)


class SystematicSearchClassifier(ConsensusSearchClassifier):
    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        scores = [(col, calc_accuracy(y, x[col], self.metric)) for col in x.columns]
        scores.sort(key=lambda tup: tup[1], reverse=METRIC_MODES[self.metric] == 'maximize')
        top_cols = [col for col, _ in scores[:cons_size]]
        return pd.Index(top_cols)


class GeneticSearchRegressor(ConsensusSearchRegressor):
    def __init__(self, cons_size=10, n_iter=200, pop_size=50, mut_prob=0.2, metric="mae", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.mut_prob = mut_prob

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        def objective(ind: Individual) -> float:
            y_pred = self._consensus_predict(x.iloc[:, list(ind)])
            return calc_accuracy(y, y_pred, self.metric)

        space = range(len(x.columns))
        task = METRIC_MODES[self.metric]
        init_cols = SystematicSearchRegressor(cons_size, self.metric).run(x, y)
        elite = Individual([x.columns.get_loc(col) for col in init_cols])

        ga = GeneticAlgorithm(task=task, pop_size=self.pop_size, crossover_prob=0.95,
                              mutation_prob=self.mut_prob, elitism=True)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=cons_size, ind_elite=elite)
        ga.run(n_iter=self.n_iter, verbose=False)
        return x.columns[list(ga.get_global_best())]


class GeneticSearchClassifier(ConsensusSearchClassifier):
    def __init__(self, cons_size=10, n_iter=200, pop_size=50, mut_prob=0.2, metric="acc", cons_size_candidates=None):
        super().__init__(cons_size, cons_size_candidates, metric)
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.mut_prob = mut_prob

    def _run_with_cons_size(self, x: DataFrame, y: Series, cons_size: int) -> Index:
        def objective(ind: Individual) -> float:
            y_pred = self._consensus_predict(x.iloc[:, list(ind)])
            return calc_accuracy(y, y_pred, self.metric)

        space = range(len(x.columns))
        task = METRIC_MODES[self.metric]
        init_cols = SystematicSearchClassifier(cons_size, self.metric).run(x, y)
        elite = Individual([x.columns.get_loc(col) for col in init_cols])

        ga = GeneticAlgorithm(task=task, pop_size=self.pop_size, crossover_prob=0.95,
                              mutation_prob=self.mut_prob, elitism=True)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=cons_size, ind_elite=elite)
        ga.run(n_iter=self.n_iter, verbose=False)
        return x.columns[list(ga.get_global_best())]
