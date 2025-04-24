import os
import datetime
import shutil
import copy
import numpy as np
import pandas as pd
import scipy
import traceback
import multiprocessing
from xgboost import XGBRegressor
from glob import glob
from joblib import Parallel, delayed

from scipy.stats import loguniform, uniform, randint
from pathlib import Path
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
from sklearn.base import clone

from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



def prepare(benchmark_collection):
    output = []
    for coll_folder in os.listdir(benchmark_collection):
        for bench_name in os.listdir(os.path.join(benchmark_collection, coll_folder)):
            output.append((benchmark_collection, coll_folder, bench_name))
    return output


def parse_data(data_path):
    data = pd.read_csv(data_path, header=None)

    smi_prop_list = []
    for smi, prop in zip(data[0], data[1]):
        smi_prop_list.append((smi, prop))

    return smi_prop_list


def combine_data(prediction_collection):
    # Iterate through the directory
    for coll_folder in os.listdir(prediction_collection):
        coll_path = os.path.join(prediction_collection, coll_folder)
        if os.path.isdir(coll_path):
            for bench_name in os.listdir(coll_path):
                bench_path = os.path.join(coll_path, bench_name)
                for pattern in ['*test.csv', '*test_default.csv', '*train.csv', '*train_default.csv']:
                    df_list = []
                    matching_files = sorted(glob(os.path.join(bench_path, pattern)))
                    for fname in matching_files:
                        df_list.append(pd.read_csv(fname))

                    # Proceed with concatenation
                    df = pd.concat(df_list, axis=1)
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
                    other_cols = [col for col in df.columns if col != 'Y_TRUE']
                    df = df[['Y_TRUE'] + other_cols]

                    # to csv
                    output_file = os.path.join(bench_path, f'combined{pattern[1:]}')
                    df.to_csv(output_file, index=False)
                    print(bench_name, pattern)


def calculate_scores(benchmark_collection, coll_folder, bench_name, descr_name, prediction_collection, hyper_parameters):

    descr_dict = {
        'scaffoldkeys': MoleculeTransformer(featurizer='scaffoldkeys', dtype=float),
        'secfp': MoleculeTransformer(featurizer='secfp', dtype=float),
        'atompair-count': MoleculeTransformer(featurizer='atompair-count', dtype=float),
        'avalon': MoleculeTransformer(featurizer='avalon', dtype=float),
        #'ecfp-count': MoleculeTransformer(featurizer='ecfp-count', dtype=float),
        'ecfp': MoleculeTransformer(featurizer='ecfp', dtype=float),
        'erg': MoleculeTransformer(featurizer='erg', dtype=float),
        'estate': MoleculeTransformer(featurizer='estate', dtype=float),
        #'fcfp-count': MoleculeTransformer(featurizer='fcfp-count', dtype=float),
        'fcfp': MoleculeTransformer(featurizer='fcfp', dtype=float),
        'maccs': MoleculeTransformer(featurizer='maccs', dtype=float),
        'pattern': MoleculeTransformer(featurizer='pattern', dtype=float),
        'rdkit': MoleculeTransformer(featurizer='rdkit', dtype=float),
        'topological-count': MoleculeTransformer(featurizer='topological-count', dtype=float),
        'topological': MoleculeTransformer(featurizer='topological', dtype=float),
        'layered': MoleculeTransformer(featurizer='layered', dtype=float),
        'erg': FPVecTransformer(kind='erg', dtype=float)}
        
    ml_dict = {
            'XGBRegressor': XGBRegressor(),
            'PLSRegression': PLSRegression(),
            'RidgeRegression': Ridge(),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'MLPRegressor': MLPRegressor(max_iter=5000, solver='adam'),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor()}

    descr_func = descr_dict[descr_name]

    n_jobs = 3

    bench_folder = os.path.join(benchmark_collection, coll_folder, bench_name)
    res_folder = os.path.join(prediction_collection, coll_folder, bench_name)
    os.makedirs(res_folder, exist_ok=True)

    # run benchmark
    res_test_default = pd.DataFrame()
    res_train_default = pd.DataFrame()
    res_test = pd.DataFrame()
    res_train = pd.DataFrame()
    skipped_df = pd.DataFrame(

    )
    # parse dataset
    data_train = parse_data(os.path.join(bench_folder, 'train.csv'))
    data_test = parse_data(os.path.join(bench_folder, 'test.csv'))

    # save true prop
    res_train['Y_TRUE'] = [i[1] for i in data_train]
    res_train_default['Y_TRUE'] = [i[1] for i in data_train]
    res_test['Y_TRUE'] = [i[1] for i in data_test]
    res_test_default['Y_TRUE'] = [i[1] for i in data_test]

    # calculate training data descriptors
    try:
        x_train = descr_func([i[0] for i in data_train])
        x_test = descr_func([i[0] for i in data_test])

        y_train = [i[1] for i in data_train]

        # scale training data descriptors
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1)

    #    print(ml_dict)

        # train machine learning model
        for method_name in ml_dict:

            print(f'({os.getpid()}): [{datetime.datetime.now()}] {bench_name, method_name, descr_name} default start', flush=True)
            
            # concat cross-validation prediction from the training set for consensus and stacking building
            model = ml_dict.get(method_name)
            y_pred = cross_val_predict(model, x_train_scaled, y_train, cv=cv, n_jobs=n_jobs)
            res_train_default[f'{descr_name}|{method_name}'] = y_pred

            # build the final 2D model
            model.fit(x_train_scaled, y_train)

            # make test set predictions
            y_pred = model.predict(x_test_scaled)
            res_test_default[f'{descr_name}|{method_name}'] = y_pred

            print(f'({os.getpid()}): [{datetime.datetime.now()}] {bench_name, method_name, descr_name} hyperopt start', flush=True)

            # make test set predictions with hyperparameters optimized
            random_search = (RandomizedSearchCV(model, hyper_parameters.get(method_name), n_iter=30, cv=cv, error_score= "raise", n_jobs=n_jobs))
            random_search.fit(x_train_scaled, y_train)
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(x_test_scaled)
            res_test[f'{descr_name}|{method_name}'] = y_pred

            # After hyperparameter optimization
            best_params = random_search.best_params_
            optimized_model = clone(model)
            optimized_model.set_params(**best_params)

            # Cross-validate with optimized hyperparameters
            y_pred = cross_val_predict(optimized_model, x_train_scaled, y_train, cv=cv, n_jobs=n_jobs)
            res_train[f'{descr_name}|{method_name}'] = y_pred

            print(f'({os.getpid()}): [{datetime.datetime.now()}] {bench_name, method_name, descr_name} hyperopt finished', flush=True)

        # to csv
        res_test.to_csv(os.path.join(res_folder, f'{descr_name}test.csv'), index=False)
        res_train_default.to_csv(os.path.join(res_folder, f'{descr_name}train_default.csv'), index=False)
        res_test_default.to_csv(os.path.join(res_folder, f'{descr_name}test_default.csv'), index=False)
        res_train.to_csv(os.path.join(res_folder, f'{descr_name}train.csv'), index=False)

        print(f'({os.getpid()}): [{datetime.datetime.now()}] {bench_name, descr_name} saved', flush=True)
    except Exception as e:
        skipped_df[bench_name] = f'{descr_name}'
        skipped_df.to_csv(os.path.join(skipped, f'{bench_name, descr_name}.csv'))


def check_exg_data(benchmark_collection, prediction_collection, descr_dict):
    #check of existing data and add missing tasks
    dirs_to_compute = []
    for coll_folder in os.listdir(benchmark_collection):
        for bench_name in os.listdir(os.path.join(benchmark_collection, coll_folder)):
            try:
                for descr_name, descr_func in descr_dict.items():
                    # The expected output file for this descriptor
                    pred_dir = os.path.join(prediction_collection, coll_folder, bench_name)
                    expected_file = os.path.join(pred_dir, f"{descr_name}test.csv")
                    if os.path.exists(expected_file):
                        print(f"{bench_name, descr_name} already computed")

                    else:
                        dirs_to_compute.append([benchmark_collection, coll_folder, bench_name, descr_name])
            except FileNotFoundError:
                for file in os.listdir(os.path.join(benchmark_collection, coll_folder, bench_name)):
                    for descr_name, descr_func in descr_dict.items():
                        dirs_to_compute.append([benchmark_collection, coll_folder, bench_name, descr_name])
    return dirs_to_compute

hyper_parameters = {
    'XGBRegressor': {
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(1e-3, 0.3),
        'n_estimators': (500, 1500)
    },
    'PLSRegression': {
        'n_components': randint(1, 11),
        'scale': [True, False],
        'tol': loguniform(1e-6, 1e-3)
    },
    'RidgeRegression': {
        'alpha': loguniform(1e-3, 100),
        'solver': ['auto'],
        'fit_intercept': [True, False],
        'max_iter': [None, 1000, 5000],
        'tol': [1e-4, 1e-3, 1e-2]
   },
    'LinearRegression': {
        'fit_intercept': [True, False],
    },
    'RandomForestRegressor': {
        'n_estimators': randint(50, 500),
        'max_depth': [None, 10, 25, 50],
        'max_features': ['sqrt', 0.3, 0.5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True]
    },
    'MLPRegressor': {
        'hidden_layer_sizes': [(50,), (100,)],
        'solver': ['adam'],
        'max_iter': [2000, 5000],
    },
    'SVR': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': [0.01, 0.1, 1]
    },
    'KNeighborsRegressor': {
        'n_neighbors': randint(1, 50),
        'weights': ['uniform', 'distance'],
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

descr_dict = {
    'scaffoldkeys': MoleculeTransformer(featurizer='scaffoldkeys', dtype=float),
    'secfp': MoleculeTransformer(featurizer='secfp', dtype=float),
    'atompair-count': MoleculeTransformer(featurizer='atompair-count', dtype=float),
    'avalon': MoleculeTransformer(featurizer='avalon', dtype=float),
    #'ecfp-count': MoleculeTransformer(featurizer='ecfp-count', dtype=float),
    'ecfp': MoleculeTransformer(featurizer='ecfp', dtype=float),
    'erg': MoleculeTransformer(featurizer='erg', dtype=float),
    'estate': MoleculeTransformer(featurizer='estate', dtype=float),
    #'fcfp-count': MoleculeTransformer(featurizer='fcfp-count', dtype=float),
    'fcfp': MoleculeTransformer(featurizer='fcfp', dtype=float),
    'maccs': MoleculeTransformer(featurizer='maccs', dtype=float),
    'pattern': MoleculeTransformer(featurizer='pattern', dtype=float),
    'rdkit': MoleculeTransformer(featurizer='rdkit', dtype=float),
    'topological-count': MoleculeTransformer(featurizer='topological-count', dtype=float),
    'topological': MoleculeTransformer(featurizer='topological', dtype=float),
    'layered': MoleculeTransformer(featurizer='layered', dtype=float),
    'erg': FPVecTransformer(kind='erg', dtype=float),

    #long
    # 'desc2D': MoleculeTransformer(featurizer='desc2D', dtype=float),
    # 'pharm2D-cats': MoleculeTransformer(featurizer=Pharmacophore2D(factory='cats'), dtype=float),
    # 'pharm2D-gobbi': MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float),
    # 'pharm2D-pmapper': MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float),
    
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

benchmark_collection =  Path(fr'../benchmark_collection_original').resolve()
prediction_collection = Path(fr'../benchmark_model_prediction').resolve()
skipped = Path(fr'../skipped').resolve()

if __name__ == "__main__":
    prediction_collection.mkdir(parents=True, exist_ok=True)
    skipped.mkdir(parents=True, exist_ok=True)

    tasks = check_exg_data(benchmark_collection, prediction_collection, descr_dict)
    Parallel(126, prefer='processes')(delayed(calculate_scores)(i, j, k, l, prediction_collection, hyper_parameters) for i, j, k, l in tasks)

    combine_data(prediction_collection)