import random

from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from qsarcons.genopt import Individual

from .genopt import GeneticAlgorithm


def calc_accuracy(y_true, y_pred, metric="mae"):
    if metric == "mae":
        acc = mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        acc = r2_score(y_true, y_pred)
    elif metric == "rmse":
        acc = root_mean_squared_error(y_true, y_pred)

    return acc


class RandomSearchRegressor:
    def __init__(self, cons_size=10, n_iter=5000, metric="mae"):
        super().__init__()

        self.cons_size = cons_size
        self.n_iter = n_iter
        self.metric = metric

    def run(self, x: DataFrame, y: Series) -> Index:

        cons_list = []
        for _ in range(self.n_iter):

            random_cons = random.sample(range(len(x.columns)), k=self.cons_size)  # skip TRUE column
            y_cons = x[x.columns[random_cons]].mean(axis=1)

            acc = calc_accuracy(y, y_cons, metric=self.metric)
            cons_list.append((random_cons, acc))
            #
            if self.metric in ["mae", "rmse"]:
                cons_list = sorted(cons_list, key=lambda x: x[1], reverse=False)  # minimize
            elif self.metric in ["r2"]:
                cons_list = sorted(cons_list, key=lambda x: x[1], reverse=True)  # maximize

            best_cons = cons_list[0][0]
            best_cons = x.columns[best_cons]

            return best_cons


class SystematicSearchRegressor:
    def __init__(self, cons_size: int = 10, metric: str = "mae") -> None:
        super().__init__()

        self.cons_size = cons_size
        self.metric = metric

    def run(self, x: DataFrame, y: Series) -> Index:

        tmp = []
        for model in x.columns:
            acc = calc_accuracy(y, x[model], metric=self.metric)
            tmp.append((model, acc))

        if self.metric in ["mae", "rmse"]:
            tmp_sorted = sorted(tmp, key=lambda x: x[1], reverse=False)  # minimize
        elif self.metric in ["r2"]:
            tmp_sorted = sorted(tmp, key=lambda x: x[1], reverse=True)  # maximize

        x_sorted = x[[i[0] for i in tmp_sorted]]

        best_cons = x_sorted.columns[: self.cons_size]

        return best_cons


class GeneticSearchRegressor:
    def __init__(self, cons_size: int = 10, n_iter: int = 200, mut_prob: float = 0.2, metric: str = "mae") -> None:
        super().__init__()

        self.cons_size = cons_size
        self.n_iter = n_iter
        self.metric = metric
        self.mut_prob = mut_prob

    def run(self, x: DataFrame, y: Series) -> Index:

        def objective(cons: Individual) -> float:
            y_cons = x[x.columns[cons]].mean(axis=1)
            acc = calc_accuracy(y, y_cons, metric=self.metric)
            return acc

        #
        space = range(len(x.columns))
        if self.metric in ["mae", "rmse"]:
            task = "minimize"
        elif self.metric in ["r2"]:
            task = "maximize"
        #
        ga = GeneticAlgorithm(task=task, pop_size=50, crossover_prob=0.8, mutation_prob=self.mut_prob, elitism=True)
        ga.set_fitness(objective)
        ga.initialize(space, ind_size=self.cons_size)
        ga.run(n_iter=200, verbose=False)
        #
        best_cons = ga.best_individual()
        best_cons = x.columns[best_cons]
        #
        return best_cons
