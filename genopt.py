import random
from random import randint
from copy import deepcopy
from statistics import pvariance, pstdev

def init_individual(space, steps):
    ind = random.sample(space, k=steps)
    return Individual(ind)


def init_population(minimax, space, pop_size, steps):
    population = Population(minimax=minimax)

    while len(population) < pop_size:
        individual = init_individual(space, steps)
        if individual not in population:
            population.append(individual)
    return population


def one_point_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    for _ in range(100):
        cut = random.randint(1, len(mother) - 1)

        sister[cut:] = father[cut:]
        brother[cut:] = mother[cut:]

        # TODO do refactoring
        if len(set(sister.container)) == len(set(brother.container)) == len(sister.container):
            break

    return sister, brother


def uniform_mutation(individual, space, prob=0):
    for _ in range(100):
        for n, gen in enumerate(individual):
            if random.random() < prob:
                individual[n] = random.choice(space)

        # TODO do refactoring
        if len(set(individual.container)) == len(individual.container):
            return individual

    return individual


def tournament_selection(pop):
    selected_inds = []
    for i in range(len(pop)):
        p1 = random.randint(0, len(pop) - 1)
        p2 = random.randint(0, len(pop) - 1)

        if pop.minimax == 'maximize':
            selected_inds.append(max(p1, p2, key=lambda x: pop[x].score))
        else:
            selected_inds.append(min(p1, p2, key=lambda x: pop[x].score))

    return selected_inds


def sigma_trunc_scaling(pop):
    pop.calc_statistics()

    c = 2

    rawAvg = pop.stats["rawAvg"]
    rawDev = pop.stats["rawDev"]

    for i in range(len(pop)):

        f = pop[i].score - rawAvg
        f += c * rawDev

        if f < 0:
            f = 0.0

        pop[i].fitness = f

    pop.calc_statistics()


def flip_coin(p):

    if p == 1.0:
        return True
    if p == 0.0:
        return False

    return True if random.random() <= p else False


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

    def evaluate(self, ):
        for ind in self:
            ind.score = self.evaluator(ind)
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


class GeneticAlgorithm:

    def __init__(self, task='minimize', pop_size=10, cross_prob=0.8, mut_prob=0.1, elitism=True, n_cpu=1):

        random.seed(42)

        self.task = task
        self.fitness_func = None
        self.population = None
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

        self.selector = tournament_selection
        self.scaler = sigma_trunc_scaling
        self.crossover = one_point_crossover
        self.mutator = uniform_mutation

        self.elitism = elitism
        self.n_cpu = n_cpu
        self.current_generation = 0

    def set_fitness(self, fitness_func):
        self.fitness_func = fitness_func

    def initialize(self, space, steps):
        self.space = space
        self.steps = steps
        self.population = init_population(self.task, self.space, self.pop_size, self.steps)
        self.population.evaluator = self.fitness_func
        self.population.scaler = self.scaler
        self.evaluate()
        self.population.scale()
        self.population.sort()
        self.population.calc_statistics()
        self.best_solution = self.best_individual()

    def evaluate(self):
        self.population.evaluate()
        return self

    def select(self):
        selected_inds = self.selector(self.population)
        return selected_inds

    def get_mating_pool(self, selected_inds):
        mating_pool = []
        for i in selected_inds:
            mating_pool.append(self.population[i])
        return mating_pool

    def crossover(self, mother, father):
        if flip_coin(self.cross_prob):
            sister, brother = self.crossover(mother, father)
        else:
            sister, brother = deepcopy(mother), deepcopy(father)
        return sister, brother

    def mutate(self, individual, space, prob):
        mutant = self.mutator(individual, space, prob=prob)
        return mutant

    def step(self):
        mating_pool = self.get_mating_pool(self.select())

        pairs = len(self.population) // 2
        new_pop = deepcopy(self.population)
        new_pop.clear()

        for i in range(pairs):
            mother = mating_pool.pop(randint(0, len(mating_pool) - 1))
            father = mating_pool.pop(randint(0, len(mating_pool) - 1))

            sister, brother = self.crossover(mother, father)

            sister_mut = self.mutate(sister, self.space, prob=self.mut_prob)
            brother_mut = self.mutate(brother, self.space, prob=self.mut_prob)

            if sister_mut not in self.population:
                new_pop.append(sister_mut)
                self.population.append(sister_mut)

            if brother_mut not in self.population:
                new_pop.append(brother_mut)
                self.population.append(brother_mut)

        while len(new_pop) < self.pop_size:
            ind = init_individual(self.space, self.steps)
            if ind not in self.population:
                new_pop.append(ind)
                self.population.append(ind)

        if len(mating_pool):
            new_pop.append(mating_pool.pop())

        new_pop.evaluate()
        new_pop.scale()
        new_pop.sort()

        if self.elitism:
            if self.task == 'maximize':
                if self.best_solution.score > new_pop.best_raw().score:
                    new_pop[-1] = self.best_solution
                else:
                    self.best_solution = new_pop.best_raw()
            else:
                if self.best_solution.score < new_pop.best_raw().score:
                    new_pop[-1] = self.best_solution
                else:
                    self.best_solution = new_pop.best_raw()

        self.population = new_pop
        self.current_generation += 1

        return self

    def run(self, n_iter=50, verbose=True):

        if verbose:
            header = ['Iter', 'rawMax', 'rawAvg', 'rawMin']
            print('{:^5}|{:^13}|{:^13}|{:^13}'.format(*header))
            print('-' * 47)
            for i in range(n_iter):
                self.step()
                self.print_stats()
        else:
            for i in range(n_iter):
                self.step()

    def best_individual(self):
        return self.population.best_raw()

    def get_statistics(self):
        self.population.calc_statistics()
        self.population.stats['Iter'] = self.current_generation
        return self.population.stats

    def print_stats(self):
        stats = self.get_statistics()
        print('{Iter:^5}|{rawMax:^13.5f}|{rawAvg:^13.5f}|{rawMin:^13.5f}'.format(**stats))

    def __repr__(self):
        pass