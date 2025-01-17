import logging

db = 'db.shelve'
target_name = 'ephedrine'
target = open(f'{target_name}.sdf', encoding='UTF-8')
steps = 5000
popsize = 50
chromo_len = 10
pop_dict_name = f'populations_{target_name}.pickle'
cpu = 10
logger = logging.getLogger("GA")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/path/to/ga.log", mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
