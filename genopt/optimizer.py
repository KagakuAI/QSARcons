from copy import deepcopy
from genopt.utils import init_random_individual
from .utils import init_population, flip_coin
from .selectors import roulette_wheel_selection
from .scalers import sigma_trunc_scaling
from .crossovers import one_point_crossover
from .mutators import uniform_mutation
from random import randint


class SGA:
    
    def __init__(self, task='minimize', pop_size=10, cross_prob=0.8, mut_prob=0.1, elitism=True, n_cpu=1):
        
        self.task = task
        self.fitness_func = None
        self.population = None
        self.pop_size = pop_size
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

        self.selector = roulette_wheel_selection
        self.scaler = sigma_trunc_scaling
        self.crossover = one_point_crossover
        self.mutator = uniform_mutation
        
        self.elitism = elitism
        self.n_cpu = n_cpu
        self.current_generation = 0

    def set_fitness(self, fitness_func):
        self.fitness_func = fitness_func

    def set_scaler_type(self, scaler_type):
        self.scaler = scaler_type

    def set_selector_type(self, selector_type):
        self.selector = selector_type

    def set_crossover_type(self, crossover_type):
        self.crossover = crossover_type
    
    def set_mutator_type(self, mutator_type):
        self.mutator = mutator_type
     
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
            ind = init_random_individual(self.space, self.steps)
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

