import random
from copy import deepcopy
from .utils import flip_coin


def uniform_mutation(individual, space, prob=0):
    for n, gen in enumerate(individual):
        if random.random() < prob:
            individual[n] = random.choice(space)
    return individual
