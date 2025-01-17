from config import *
from genopt.crossovers import two_point_crossover
from genopt.mutators import uniform_mutation
from genopt.optimizer import SGA
from genopt.selectors import tournament_selection
from genopt.scalers import sigma_trunc_scaling
from genopt.utils import weighted_sample_without_replacement
from CGRtools.files import SDFRead
import logging
import os
from pickle import dump, load
from time import time

os.environ['DB'] = db
os.environ['DATA'] = '/home/adelia/rules'

from RNNSynthesis.environment import SimpleSynthesis


target = next(SDFRead(target))
logger.info(f" Program started. Target: {target_name}, {str(target)}, Steps: {steps}, "
            f"Population size: {popsize}, Number of steps: {chromo_len}")

play = SimpleSynthesis(target, steps=10 ** 6)

ga = SGA(task='maximize', pop_size=popsize, cross_prob=0.9, mut_prob=0.1, elitism=True, n_cpu=cpu)
ga.set_selector_type(tournament_selection)
ga.set_scaler_type(sigma_trunc_scaling)
ga.set_crossover_type(two_point_crossover)
ga.set_mutator_type(uniform_mutation)


def fitness(chromo):
    start = time()
    last_reward = 0
    play.reset()
    for step in chromo:
        state, reward, done, info = play.step(int(step))
        if state:
            if len(state) - len(target) >= 20:
                break
            last_reward = reward
            if done:
                if play.depth < 5 and not play.stop:
                    logger.info('DONE. Target synthesized!')
                else:
                    logger.info('DONE')
                logger.info(f'current state: {state}, reward {reward}, info {info}\n')
                break

    print(str(time() - start))
    return last_reward


ga.set_fitness(fitness)

pop_dict = {}
# init action space and population
ga.initialize(play.action_space, chromo_len)
pop_dict['init'] = ga.population
with open(pop_dict_name, 'wb') as f:
    dump(pop_dict, f)

tanimoto = 0

previous_pop = load(open(pop_dict_name, 'rb')).popitem()[1]
ga.population = previous_pop
ga.step()
for i in range(1, steps + 1):
    logger.info(f'step {i} started')
    ga.step()
    pop_dict[i] = ga.population
    with open(pop_dict_name, 'wb') as f:
        dump(pop_dict, f)
    chromo = ga.best_individual()
    stat = ga.population.calc_statistics()
    stat.stats['chromo'] = ' '.join(map(str, chromo))
    tanimoto = chromo.score
    for k in stat.stats:
        logger.info(f'{k}:, {stat.stats[k]}')
    if tanimoto >= 10:
        logger.info('Target synthesized')
        for r in play.render():
            logger.info(r)
        break

for r in play.render():
    logger.info(r)
