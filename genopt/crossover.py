import random
from copy import deepcopy
from .utils import flip_coin


def one_point_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    cut = random.randint(1, len(mother) - 1)

    sister[cut:] = father[cut:]
    brother[cut:] = mother[cut:]

    return sister, brother


def two_point_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    cut1 = random.randint(1, len(mother) - 1)
    cut2 = random.randint(1, len(mother) - 1)

    if cut1 == cut2:
        cut2 = random.randint(1, len(mother) - 1)

    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    sister[0:cut1], sister[cut2:] = father[0:cut1], father[cut2:]
    brother[0:cut1], brother[cut2:] = mother[0:cut1], mother[cut2:]

    return sister, brother


def uniform_point_crossover(mother, father):
    sister = deepcopy(mother)
    brother = deepcopy(father)

    for i, (gs, gb) in enumerate(zip(sister, brother)):
        if flip_coin(0.5):
            sister[i] = gb
            brother[i] = gs
        elif flip_coin(0.25):
            sister[random.randint(0, len(sister) - 1)] = gb
            brother[random.randint(0, len(sister) - 1)] = gs

    return sister, brother

