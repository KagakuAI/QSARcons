import random
from copy import deepcopy
from random import randint
from statistics import pstdev, pvariance
from typing import Callable, List, Tuple, Union, Iterator

class Individual:

    def __init__(self, container: List[int]) -> None:
        self.container = container
        self.score = 0
        self.fitness = 0
        self.rank = 0

    def __getitem__(self, item: slice) -> List[int]:
        return self.container[item]

    def __setitem__(self, key: Union[int, slice], value: Union[List[int], int]) -> None:
        self.container[key] = value

    def __delitem__(self, key):
        del self.container[key]

    def __iter__(self) -> Iterator:
        return iter(self.container)

    def __len__(self) -> int:
        return len(self.container)

    def __eq__(self, other: "Individual") -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.container)))

    def __repr__(self):
        return repr(self.container)


class Population:

    def __init__(self, task: str = "minimize") -> None:
        self.container = []
        self.task = task
        self.evaluator = None
        self.stats = {}

    def __len__(self) -> int:
        return len(self.container)

    def __getitem__(self, item: int) -> Individual:
        return self.container[item]

    def __setitem__(self, index: int, value: Individual) -> None:
        self.container[index] = value

    def __iter__(self) -> Iterator:
        return iter(self.container)

    def __repr__(self):
        return repr(self.container)

    def append(self, individual: Individual) -> None:
        self.container.append(individual)

    def clear(self) -> None:
        self.container.clear()

    def evaluate(self, ) -> "Population":
        for ind in self:
            ind.score = self.evaluator(ind)
        return self


    def rank(self):
        for rank, ind in enumerate(reversed(self), 1):
            ind.rank = rank

    def sort(self) -> "Population":
        if self.task == "maximize":
            self.container.sort(key=key_raw_score, reverse=True)
        else:
            self.container.sort(key=key_raw_score)
        return self

    def best_score(self) -> Individual:
        if self.task == "maximize":
            return max(self, key=key_raw_score)
        else:
            return min(self, key=key_raw_score)

    def best_fitness(self):
        if self.task == "maximize":
            return max(self, key=key_fitness_score)
        else:
            return min(self, key=key_fitness_score)

    def calc_stat(self) -> "Population":

        sum_score = sum(self[i].score for i in range(len(self)))

        self.stats.update(
            {
                "max_score": max(self, key=key_raw_score).score,
                "min_score": min(self, key=key_raw_score).score,
                "mean_score": sum_score / len(self),
                "var_score": pvariance(ind.score for ind in self),
                "dev_score": pstdev(ind.score for ind in self),
                "diversity": None,
            }
        )

        if self.best_score().fitness is not None:
            fit_sum = sum(self[i].fitness for i in range(len(self)))

            self.stats.update(
                {
                    "max_fitness": max(self, key=key_fitness_score).fitness,
                    "min_fitness": min(self, key=key_fitness_score).fitness,
                    "mean_fitness": fit_sum / len(self),
                }
            )

        return self


class GeneticAlgorithm:

    def __init__(
        self,
        task: str = "minimize",
        pop_size: int = 10,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        elitism: bool = True,
        random_seed=42,
    ) -> None:

        self.task = task
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_seed = random_seed

        # genetic operators
        self.selector = tournament_selection
        self.pair_crossover = one_point_crossover
        self.mutator = uniform_mutation

        self.elitism = elitism
        self.current_generation = 0
        self.best_individuals = []

        random.seed(self.random_seed)

    def __repr__(self):
        return f"<GeneticAlgorithm gen={self.current_generation} pop_size={self.pop_size}>"

    def set_fitness(self, fitness_func: Callable) -> None:
        self.fitness = fitness_func

    def initialize(self, ind_space: range, ind_size: int) -> None:

        # individual size and parameters
        self.ind_space = ind_space
        self.ind_size = ind_size

        # create zero population
        self.population = init_population(
            task=self.task, pop_size=self.pop_size, ind_space=self.ind_space, ind_size=self.ind_size
        )

        self.population.evaluator = self.fitness

        self.evaluate()
        self.population.sort()
        self.population.calc_stat()
        self.best_solution = self.best_individual()

    def evaluate(self) -> "GeneticAlgorithm":
        self.population.evaluate()
        return self

    def select(self) -> List[Individual]:
        return [self.population[i] for i in self.selector(self.population)]

    def crossover(self, mother, father):

        if random.random() <= self.crossover_prob:
            sister, brother = self.pair_crossover(mother, father)
        else:
            sister, brother = deepcopy(mother), deepcopy(father)
        return sister, brother

    def mutate(self, individual: Individual, space: range, prob: float) -> Individual:
        mutant = self.mutator(individual, space, prob=prob)
        return mutant

    def get_global_best(self):
        if self.task == "maximize":
            return max(self.best_individuals, key=lambda x: x.score)
        else:
            return min(self.best_individuals, key=lambda x: x.score)

    def step(self) -> "GeneticAlgorithm":

        new_population = deepcopy(self.population)
        new_population.clear()

        mating_pool = self.select()
        num_pair = len(self.population) // 2
        for i in range(num_pair):

            mother = mating_pool.pop(randint(0, len(mating_pool) - 1))
            father = mating_pool.pop(randint(0, len(mating_pool) - 1))

            sister, brother = self.pair_crossover(mother, father)

            sister_mutated = self.mutate(sister, self.ind_space, prob=self.mutation_prob)
            brother_mutated = self.mutate(brother, self.ind_space, prob=self.mutation_prob)

            if sister_mutated not in self.population:
                new_population.append(sister_mutated)
                self.population.append(sister_mutated)

            if brother_mutated not in self.population:
                new_population.append(brother_mutated)
                self.population.append(brother_mutated)

        while len(new_population) < self.pop_size:
            ind = init_individual(self.ind_space, self.ind_size)
            if ind not in self.population:
                new_population.append(ind)
                self.population.append(ind)

        if len(mating_pool):
            new_population.append(mating_pool.pop())

        new_population.evaluate()
        new_population.sort()
        self.best_individuals.append(deepcopy(new_population.best_score()))

        if self.elitism:
            if self.task == "maximize":
                if self.best_solution.score > new_population.best_score().score:
                    new_population[-1] = self.best_solution
                else:
                    self.best_solution = new_population.best_score()
            else:
                if self.best_solution.score < new_population.best_score().score:
                    new_population[-1] = self.best_solution
                else:
                    self.best_solution = new_population.best_score()

        self.population = new_population
        self.current_generation += 1

        return self

    def run(self, n_iter: int = 50, verbose: bool = False) -> None:
        for i in range(n_iter):
            self.step()

    def best_individual(self) -> Individual:
        return self.population.best_score()


def init_individual(ind_space: range = None, ind_size: int = None) -> Individual:
    """Initializes random individual with size ind_size sampling from ind_space
    (optimized parameters)

    :param ind_space: The space where optimized parameters (genes) are defined
    :param ind_size: The size/length of the individual (the number of optimized parameters)
    :return: The Individual object
    """
    ind = Individual(random.sample(ind_space, k=ind_size))
    return ind


def init_population(
    task: str = None, pop_size: int = None, ind_space: range = None, ind_size: int = None) -> Population:
    """Initializes random population of size pop_size.

    :param task: The optimization type (minimize or maximize)
    :param pop_size: The size of the population
    :param ind_space: The space where optimized parameters (genes) are
        defined
    :param ind_size: The size/length of the individual (the number of
        optimized parameters)
    :return: Population with size pop_size
    """

    population = Population(task=task)

    while len(population) < pop_size:
        individual = init_individual(ind_space, ind_size)
        if individual not in population:
            population.append(individual)
    return population


def one_point_crossover(mother: Individual, father: Individual) -> Tuple[Individual, Individual]:
    """Randomly select a point along the chromosome (individual) and swap all
    genes after the crossover point between the two parents to create two new
    offspring (individual).

    :param mother: The selected individual 1 (mother)
    :param father: The selected individual 2 (father)
    :return: The two newly generated individuals (sister and brother)
    """

    sister = deepcopy(mother)
    brother = deepcopy(father)

    for _ in range(100):
        cut = random.randint(1, len(mother) - 1)

        sister[cut:] = father[cut:]
        brother[cut:] = mother[cut:]

        # prevent gene repetition
        if len(set(sister.container)) == len(set(brother.container)) == len(sister.container):
            break

    return sister, brother


def uniform_mutation(individual: Individual, ind_space: range, prob: float = 0) -> Individual:
    """Iterate over each gene, and if the generated probability is lower than
    the predefined probability of mutation, replace the gene with another gene
    randomly sampled from ind_space.

    :param individual: the selected individual
    :param ind_space: The space where optimized parameters (genes) are
        defined
    :param prob: The probability of mutation
    :return: The mutated individual
    """
    mutant = deepcopy(individual)
    for _ in range(100):
        for n, gen in enumerate(mutant):
            if random.random() < prob:
                mutant[n] = random.choice(ind_space)

        # prevent repeating enes
        if len(set(mutant.container)) == len(mutant.container):
            return mutant

    return mutant


def tournament_selection(population: Population) -> List[int]:
    """Selects randomly two individuals and compare them by their scores. The
    individual with the better score is selected for the next generation. This
    procedure repeats as many times as the population size. In the current
    implementation same individual can be sampled for a tournament more than 1
    time, or can be compared to itself. It means that the same individual can
    participate in breeding several times or even breed to itself.

    :param population: The current population
    :return: The indexes of the selected individuals
    """

    selected = []
    for _ in range(len(population)):
        competitors = random.sample(range(len(population)), 2)
        if population.task == "maximize":
            winner = max(competitors, key=lambda i: population[i].score)
        else:
            winner = min(competitors, key=lambda i: population[i].score)
        selected.append(winner)
    return selected


def key_raw_score(individual: Individual) -> float:
    return individual.score


def key_fitness_score(individual: Individual) -> float:
    return individual.fitness