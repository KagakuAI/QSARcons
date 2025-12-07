import random
from copy import deepcopy
from random import randint
from statistics import pstdev, pvariance
from typing import Callable, List, Tuple, Union, Iterator


class Individual:
    """
    Represents a single solution (chromosome) in a genetic algorithm.

    Each individual consists of a list of genes (integers or categorical values),
    which can be mutated, crossed over, and evaluated for fitness.

    Attributes
    ----------
    container : List[int]
        The list of genes that define this individual.
    score : float
        The raw evaluation score of the individual (objective function value).
    fitness : float
        The normalized fitness value used for selection.
    rank : int
        The individual's rank within the current population after sorting.
    """

    def __init__(self, container: List[int]) -> None:
        self.container = container
        self.score = 0
        self.fitness = 0
        self.rank = 0

    def __getitem__(self, item: slice) -> List[int]:
        """Allow slicing access to the genes."""
        return self.container[item]

    def __setitem__(self, key: Union[int, slice], value: Union[List[int], int]) -> None:
        """Allow item or slice assignment for genes."""
        self.container[key] = value

    def __delitem__(self, key):
        """Delete a gene or slice of genes."""
        del self.container[key]

    def __iter__(self) -> Iterator:
        """Enable iteration over genes."""
        return iter(self.container)

    def __len__(self) -> int:
        """Return the number of genes."""
        return len(self.container)

    def __eq__(self, other: "Individual") -> bool:
        """Equality check based on the hash of the gene set."""
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        """Generate a hash based on sorted genes to ensure uniqueness."""
        return hash(tuple(sorted(self.container)))

    def __repr__(self):
        """String representation of the individual (its gene sequence)."""
        return repr(self.container)


class Population:
    """
    Represents a collection of individuals in a genetic algorithm.

    The population handles evaluation, sorting, statistics, and ranking.
    """

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
        """Add a new individual to the population."""
        self.container.append(individual)

    def clear(self) -> None:
        """Remove all individuals from the population."""
        self.container.clear()

    def evaluate(self) -> "Population":
        """
        Evaluate all individuals in the population using the assigned evaluator.
        """
        for ind in self:
            ind.score = self.evaluator(ind)
        return self

    def rank(self):
        """Assign rank numbers to individuals after sorting."""
        for rank, ind in enumerate(reversed(self), 1):
            ind.rank = rank

    def sort(self) -> "Population":
        """Sort the population by score according to the optimization task."""
        if self.task == "maximize":
            self.container.sort(key=key_raw_score, reverse=True)
        else:
            self.container.sort(key=key_raw_score)
        return self

    def best_score(self) -> Individual:
        """Return the best individual based on raw score."""
        if self.task == "maximize":
            return max(self, key=key_raw_score)
        else:
            return min(self, key=key_raw_score)

    def best_fitness(self):
        """Return the best individual based on fitness value."""
        if self.task == "maximize":
            return max(self, key=key_fitness_score)
        else:
            return min(self, key=key_fitness_score)

    def calc_stat(self) -> "Population":
        """
        Compute descriptive statistics for the population:
        - max/min/mean score
        - variance and standard deviation
        - optional fitness metrics if available
        """
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
    """
    Core implementation of a Genetic Algorithm (GA).

    Supports population initialization, selection, crossover, mutation,
    elitism, and multi-generation optimization.
    """

    def __init__(
        self,
        task: str = "minimize",
        pop_size: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
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
        """Assign the fitness function used for individual evaluation."""
        self.fitness = fitness_func

    def initialize(self, ind_space: range, ind_size: int) -> None:
        """
        Create an initial population of random individuals and evaluate them.
        """
        self.ind_space = ind_space
        self.ind_size = ind_size
        self.population = init_population(
            task=self.task, pop_size=self.pop_size, ind_space=self.ind_space, ind_size=self.ind_size
        )
        self.population.evaluator = self.fitness
        self.evaluate()
        self.population.sort()
        self.population.calc_stat()
        self.best_solution = self.best_individual()

    def evaluate(self) -> "GeneticAlgorithm":
        """Evaluate the current population."""
        self.population.evaluate()
        return self

    def select(self) -> List[Individual]:
        """Perform selection based on the defined selection operator."""
        return [self.population[i] for i in self.selector(self.population)]

    def crossover(self, mother, father):
        """Perform crossover between two individuals."""
        if random.random() <= self.crossover_prob:
            sister, brother = self.pair_crossover(mother, father)
        else:
            sister, brother = deepcopy(mother), deepcopy(father)
        return sister, brother

    def mutate(self, individual: Individual, space: range, prob: float) -> Individual:
        """Apply mutation to an individual."""
        mutant = self.mutator(individual, space, prob=prob)
        return mutant

    def get_global_best(self):
        """Return the best individual found across all generations."""
        if self.task == "maximize":
            return max(self.best_individuals, key=lambda x: x.score)
        else:
            return min(self.best_individuals, key=lambda x: x.score)

    def step(self) -> "GeneticAlgorithm":
        """
        Execute one iteration (generation) of the genetic algorithm."""
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

    def run(self, n_iter: int = 50) -> None:
        """Run the genetic algorithm for a specified number of generations."""
        for i in range(n_iter):
            self.step()

    def best_individual(self) -> Individual:
        """Return the best individual in the current population."""
        return self.population.best_score()


def init_individual(ind_space: range = None, ind_size: int = None) -> Individual:
    """Create a random individual by sampling unique genes from the search space."""
    ind = Individual(random.sample(ind_space, k=ind_size))
    return ind


def init_population(task: str = None, pop_size: int = None, ind_space: range = None, ind_size: int = None) -> Population:
    """Generate a random population of unique individuals."""
    population = Population(task=task)
    while len(population) < pop_size:
        individual = init_individual(ind_space, ind_size)
        if individual not in population:
            population.append(individual)
    return population


def one_point_crossover(mother: Individual, father: Individual) -> Tuple[Individual, Individual]:
    """Perform one-point crossover ensuring no duplicate genes in offspring."""
    sister = deepcopy(mother)
    brother = deepcopy(father)
    for _ in range(100):
        cut = random.randint(1, len(mother) - 1)
        sister[cut:] = father[cut:]
        brother[cut:] = mother[cut:]
        if len(set(sister.container)) == len(set(brother.container)) == len(sister.container):
            break
    return sister, brother


def uniform_mutation(individual: Individual, ind_space: range, prob: float = 0) -> Individual:
    """Mutate genes with uniform probability while maintaining gene uniqueness."""
    mutant = deepcopy(individual)
    for _ in range(100):
        for n, gen in enumerate(mutant):
            if random.random() < prob:
                mutant[n] = random.choice(ind_space)
        if len(set(mutant.container)) == len(mutant.container):
            return mutant
    return mutant


def tournament_selection(population: Population) -> List[int]:
    """Perform tournament selection based on individual scores."""
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
    """Return an individual's raw score (used as sort key)."""
    return individual.score


def key_fitness_score(individual: Individual) -> float:
    """Return an individual's fitness score (used as sort key)."""
    return individual.fitness
