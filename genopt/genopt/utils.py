import os
import csv
import random
from datetime import datetime
from .population import Individual, Population
import matplotlib.pyplot as plt


def init_random_individual(space, steps):
    ind = []
    for i in range(steps):
        ind.append(random.choice(space))
    return Individual(ind)


def init_population(minimax, space, pop_size, steps):
    population = Population(minimax=minimax)

    while len(population) < pop_size:
        individual = init_random_individual(space, steps)
        if individual not in population:
            population.append(individual)
    return population


def weighted_sample_without_replacement(population, weights, k=1):
    weights = list(weights)
    positions = range(len(population))
    indices = []
    n = 0
    while True:
        n += 1
        if n > 100:
            return [population[0], population[1], population[2]]
        needed = k - len(indices)
        if not needed:
            break
        for i in random.choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return [population[i] for i in indices]


def flip_coin(p):

    if p == 1.0:
        return True
    if p == 0.0:
        return False

    return True if random.random() <= p else False
    
    
class DumpStats:
    
    def __init__(self, path):
        
        self.tmp = []
        
        self.columns = ['Iter', 'rawMax', 'rawMin', 'rawAvg', 'rawVar', 
                        'rawDev', 'diversity', 'fitMax', 'fitMin', 'fitAvg']
        
        data = '/{}__genalg_opt/'.format(datetime.strftime(datetime.now(), '%d_%m_%Y__%H_%M_%S'))
        self.filename = path + data + '/ga_stats'
        
        if not os.path.exists(os.path.dirname(self.filename)):
            dir_name = os.path.dirname(self.filename)
            os.makedirs(dir_name)
    
        with open(self.filename + '.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns, delimiter=';')
            writer.writeheader()
            
    def dump_raw(self, raw):
        
        self.tmp.append(raw)
        
        with open(self.filename + '.csv', 'a', newline='') as file:
            
            writer = csv.DictWriter(file, fieldnames=self.columns, delimiter=';')
            writer.writerow(raw)
    
    def save_graphs(self):
        
        n_iter = [i['Iter'] for i in self.tmp]
        rawMax = [i['rawMax'] for i in self.tmp]
        rawMin = [i['rawMin'] for i in self.tmp]
        rawAvg = [i['rawAvg'] for i in self.tmp]
        
        plt.figure(figsize=(20, 10))

        plt.plot(n_iter, rawMax, label='MAX')
        plt.plot(n_iter, rawAvg, label='AVG')
        plt.plot(n_iter, rawMin, label='MIN')

        plt.xlabel('Generations', fontsize=18)
        plt.ylabel('Score', fontsize=18)
        plt.title('Simple GA', fontsize=18)
        
        plt.axis([0, max(n_iter), 0, max(rawMax)])

        plt.legend(fontsize=18)
        
        plt.savefig(self.filename + '.png')
        plt.close()
        
        return self
