from multiprocessing import Pool
from statistics import pvariance, pstdev


def key_raw_score(individual):
    return individual.score


def key_fitness_score(individual):
    return individual.fitness


class Individual:

    def __init__(self, container):
        self.container = container
        self.score = 0
        self.fitness = 0
        self.rank = 0

    def __getitem__(self, item):
        return self.container[item]

    def __setitem__(self, key, value):
        self.container[key] = value

    def __delitem__(self, key):
        del self.container[key]

    def __iter__(self):
        return iter(self.container)

    def __len__(self):
        return len(self.container)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(tuple(sorted(self.container)))

    def getslice(self, a, b, step=1):
        return dict(list(self.container.items())[a:b:step])

    def update(self, other):
        self.container.update(other)

    def gens(self):
        return self.container.keys()

    def __repr__(self):
        return repr(self.container)


class Population:

    def __init__(self, minimax='minimize'):
        self.container = []
        self.minimax = minimax
        self.evaluator = None
        self.scaler = None
        self.stats = {}

    def __len__(self):
        return len(self.container)

    def __getitem__(self, item):
        return self.container[item]

    def __setitem__(self, index, value):
        self.container[index] = value

    def __iter__(self):
        return iter(self.container)

    def __repr__(self):
        return repr(self.container)

    def append(self, individual):
        self.container.append(individual)

    def clear(self):
        self.container.clear()

    def evaluate(self, n_cpu=1):
        pool = Pool(n_cpu)
        scores = pool.map(self.evaluator, self)
        for ind, score in zip(self, scores):
            ind.score = score
        pool.close()
        return self

    def scale(self):
        self.scaler(self)

    def ranking(self):
        for rank, ind in enumerate(reversed(self), 1):
            ind.rank = rank

    def sort(self):
        if self.minimax == 'maximize':
            self.container.sort(key=key_raw_score, reverse=True)
        else:
            self.container.sort(key=key_raw_score)
        return self

    def best_raw(self):
        if self.minimax == 'maximize':
            return max(self, key=key_raw_score)
        else:
            return min(self, key=key_raw_score)

    def best_fitness(self):
        if self.minimax == 'maximize':
            return max(self, key=key_fitness_score)
        else:
            return min(self, key=key_fitness_score)

    def calc_statistics(self):

        raw_sum = sum(self[i].score for i in range(len(self)))

        self.stats.update({'rawMax': max(self, key=key_raw_score).score,
                           'rawMin': min(self, key=key_raw_score).score,
                           'rawAvg': raw_sum / len(self),
                           'rawVar': pvariance(ind.score for ind in self),
                           'rawDev': pstdev(ind.score for ind in self),
                           'diversity': None
                           })

        if self.best_raw().fitness is not None:
            fit_sum = sum(self[i].fitness for i in range(len(self)))

            self.stats.update({'fitMax': max(self, key=key_fitness_score).fitness,
                               'fitMin': min(self, key=key_fitness_score).fitness,
                               'fitAvg': fit_sum / len(self)
                               })

        return self
