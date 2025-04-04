import os
import shutil
import numpy as np
import pandas as pd
import scipy
from xgboost import XGBRegressor

from scipy.stats import loguniform, uniform, randint
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from rdkit import Chem

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import MinMaxScaler

from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

# #additional descriptors
# from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
# import dgl
# import dgllife
# from molfeat.trans.pretrained import GraphormerTransformer
# from molfeat.trans.pretrained import PretrainedDGLTransformer


def parse_data(data_path):
    data = pd.read_csv(data_path, header=None)
    
    smi_prop_list = []
    for smi, prop in zip(data[0], data[1]):
        smi_prop_list.append((smi, prop))

    return smi_prop_list


def process(benchmark_collection, coll_folder, bench_name, prediction_collection, descr_names, ml_dict, hyper_parameters):

    # benchmark dataset
    bench_folder = os.path.join(benchmark_collection, coll_folder, bench_name)
    res_folder = os.path.join(prediction_collection, coll_folder, bench_name)
    os.makedirs(res_folder, exist_ok=True)

    # run benchmark 
    res_test = pd.DataFrame()
    res_train = pd.DataFrame()

    # parse dataset
    data_train = parse_data(os.path.join(bench_folder, 'train.csv'))
    data_test = parse_data(os.path.join(bench_folder, 'test.csv'))
    
    # save true prop
    res_train['Y_TRUE'] = [i[1] for i in data_train]
    res_test['Y_TRUE'] = [i[1] for i in data_test]
    
    # calc 2D descriptors
    for descr_name in descr_names:
        descr_func = descr_dict.get(descr_name)
        # calculate training data descriptors
        try:
            x_train = descr_func([i[0] for i in data_train])
            x_test = descr_func([i[0] for i in data_test])
        except:
            continue
        
        y_train = [i[1] for i in data_train]

        # scale training data descriptors
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)

        # train machine learning model
        for method_name in ml_dict:
            model = ml_dict.get(method_name)

            # concat cross-validation prediction from the training set for consensus and stacking building
            y_pred = cross_val_predict(model, x_train_scaled, y_train, cv=cv, n_jobs=1)
            res_train[f'{descr_name}|{method_name}'] = y_pred
            res_train.to_csv(os.path.join(res_folder, f'{bench_name}_traincv.csv'), index=False)

            # build the final 2D model
            model.fit(x_train_scaled, y_train)
            
            # make test set predictions
            y_pred = model.predict(x_test_scaled)
            res_test[f'{descr_name}|{method_name}'] = y_pred

            # make test set predictions with hyperparameters optimized
            model = (RandomizedSearchCV(model, hyper_parameters.get(method_name), n_iter=10, cv=5, error_score='raise', n_jobs=1))
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            res_test[f'{descr_name}|{method_name}_RSCV'] = y_pred

            res_test.to_csv(os.path.join(res_folder, f'{bench_name}_test.csv'), index=False)


def prepare(benchmark_collection):
    output = []
    for coll_folder in os.listdir(benchmark_collection):
        for bench_name in os.listdir(os.path.join(benchmark_collection, coll_folder)):
            output.append((benchmark_collection, coll_folder, bench_name))
    return output


descr_dict = {
    'scaffoldkeys': MoleculeTransformer(featurizer='scaffoldkeys', dtype=float),
    'secfp': MoleculeTransformer(featurizer='secfp', dtype=float),
    'atompair-count': MoleculeTransformer(featurizer='atompair-count', dtype=float),
    'avalon': MoleculeTransformer(featurizer='avalon', dtype=float),
    'ecfp-count': MoleculeTransformer(featurizer='ecfp-count', dtype=float),
    'ecfp': MoleculeTransformer(featurizer='ecfp', dtype=float),
    'erg': MoleculeTransformer(featurizer='erg', dtype=float),
    'estate': MoleculeTransformer(featurizer='estate', dtype=float),
    'fcfp-count': MoleculeTransformer(featurizer='fcfp-count', dtype=float),
    'fcfp': MoleculeTransformer(featurizer='fcfp', dtype=float),
    'maccs': MoleculeTransformer(featurizer='maccs', dtype=float),
    'pattern': MoleculeTransformer(featurizer='pattern', dtype=float),
    'rdkit': MoleculeTransformer(featurizer='rdkit', dtype=float),
    'topological-count': MoleculeTransformer(featurizer='topological-count', dtype=float),
    'topological': MoleculeTransformer(featurizer='topological', dtype=float),
    'layered': MoleculeTransformer(featurizer='layered', dtype=float),
    'erg': FPVecTransformer(kind='erg', dtype=float),

    #long
    'desc2D': MoleculeTransformer(featurizer='desc2D', dtype=float),
    'pharm2D-cats': MoleculeTransformer(featurizer=Pharmacophore2D(factory='cats'), dtype=float),
    'pharm2D-gobbi': MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float),
    'pharm2D-pmapper': MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float),
    
    # Need additional dependencies
    # 'map4': MoleculeTransformer(featurizer='map4', dtype=float),

    # 'ChemGPT-4.7M': PretrainedHFTransformer(kind='ChemGPT-4.7M', notation='selfies', dtype=float), #takes time
    # 'MolT5': PretrainedHFTransformer(kind='MolT5', notation='smiles', dtype=float),
    # 'ChemBERTa-77M-MLM': PretrainedHFTransformer(kind='ChemBERTa-77M-MLM', notation='smiles', dtype=float),
    # 'ChemGPT-1.2B': PretrainedHFTransformer(kind='ChemGPT-1.2B', notation='selfies', dtype=float), #takes time
    # 'GPT2-Zinc480M-87M': PretrainedHFTransformer(kind='GPT2-Zinc480M-87M', notation='smiles', dtype=float),
    # 'ChemBERTa-77M-MTR': PretrainedHFTransformer(kind='ChemBERTa-77M-MTR', notation='smiles', dtype=float),
    # 'Roberta-Zinc480M-102M': PretrainedHFTransformer(kind='Roberta-Zinc480M-102M', notation='smiles', dtype=float),
    # 'gin_supervised_contextpred': PretrainedDGLTransformer(kind='gin_supervised_contextpred', dtype=float),
    # 'jtvae_zinc_no_kl': PretrainedDGLTransformer(kind='jtvae_zinc_no_kl', dtype=float),
    # 'gin_supervised_edgepred': PretrainedDGLTransformer(kind='gin_supervised_edgepred', dtype=float),
    # 'gin_supervised_masking': PretrainedDGLTransformer(kind='gin_supervised_masking', dtype=float),
}


hyper_parameters = {
    'XGBRegressor': {
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(1e-3, 0.3),
        'n_estimators': randint(100, 1000),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': loguniform(1e-6, 1),
        'min_child_weight': randint(1, 8),
        'reg_alpha': loguniform(1e-3, 100),
        'reg_lambda': loguniform(1e-3, 100)
    },
    'PLSRegression': {
        'n_components': randint(1, 11),
        'scale': [True, False],
#        'max_iter': [500, 1000, 2000],
        'tol': loguniform(1e-6, 1e-3)
    },
    'RidgeRegression': {
        'alpha': loguniform(1e-3, 100),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
        'fit_intercept': [True, False],
        'max_iter': [None, 1000, 5000],
        'tol': [1e-4, 1e-3, 1e-2]
    },
    'LinearRegression': {
        'fit_intercept': [True, False],
        'positive': [True, False]
    },
    'RandomForestRegressor': {
        'n_estimators': randint(50, 500),
        'max_depth': [None, 10, 25, 50],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True]
    },
    'MLPRegressor': {
        'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
        'activation': ['relu', 'tanh', 'logistic'],
#        'solver': ['adam', 'sgd', 'lbfgs'],
        'solver': ['sgd'],
        'alpha': loguniform(1e-5, 1e-1),
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [2000, 5000]
    },
    'SVR': {
        'C': loguniform(1e-3, 1e3),
        'epsilon': uniform(0.01, 0.3),
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': loguniform(1e-5, 1),
#        'degree': [2, 3]
    },
    'KNeighborsRegressor': {
        'n_neighbors': randint(1, 50),
        'weights': ['uniform', 'distance'],
#        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#        'p': [1, 2],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }
}

ml_dict = {
            'XGBRegressor': XGBRegressor(),
            'PLSRegression': PLSRegression(),
            'RidgeRegression': Ridge(),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'MLPRegressor': MLPRegressor(max_iter=5000, solver='adam'),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor()
}

# input data
benchmark_collection =  Path('../benchmark_collection_original').resolve()

# output data and calculations
if __name__ == '__main__':
    prediction_collection = Path('../benchmark_model_prediction').resolve()

    if prediction_collection.exists():
        shutil.rmtree(prediction_collection)

    descr_names = list(descr_dict.keys())
    ml_funct = list(ml_dict.keys())

    with Pool(cpu_count() - 1) as pool:
            list(pool.starmap(partial(process, 
                                prediction_collection=prediction_collection, 
                                descr_names=descr_names, 
                                ml_dict=ml_dict,
                                hyper_parameters=hyper_parameters),
                                    prepare(benchmark_collection)))
