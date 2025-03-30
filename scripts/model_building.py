import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from rdkit import Chem
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler


from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D


def parse_data(data_path):
    data = pd.read_csv(data_path, header=None)
    
    smi_prop_list = []
    for smi, prop in zip(data[0], data[1]):
        smi_prop_list.append((smi, prop))

    return smi_prop_list


def process(benchmark_collection, coll_folder, bench_name, prediction_collection, descr_list, ml_list):

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
    for descr_func, descr_name in descr_list:

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
        for model, method_name in ml_list:

            # concat cross-validation prediction from the training set for consensus and stacking building
            y_pred = cross_val_predict(model, x_train_scaled, y_train, cv=cv, n_jobs=1)
            res_train[f'{descr_name}|{method_name}'] = y_pred
            res_train.to_csv(os.path.join(res_folder, f'{bench_name}_traincv.csv'), index=False)

            # build the final 2D model
            model.fit(x_train_scaled, y_train)
            
            # make test set predictions
            y_pred = model.predict(x_test_scaled)
            res_test[f'{descr_name}|{method_name}'] = y_pred
            res_test.to_csv(os.path.join(res_folder, f'{bench_name}_test.csv'), index=False)


def prepare(benchmark_collection):
    output = []
    for coll_folder in os.listdir(benchmark_collection):
        for bench_name in os.listdir(os.path.join(benchmark_collection, coll_folder)):
            output.append((benchmark_collection, coll_folder, bench_name))
    return output


descr_list = [
                (MoleculeTransformer(featurizer='cats2d', dtype=float), "cats2d"), # fails sometimes
                (MoleculeTransformer(featurizer='scaffoldkeys', dtype=float), "scaffoldkeys"),
                (MoleculeTransformer(featurizer='secfp', dtype=float), "secfp"),
                (MoleculeTransformer(featurizer='atompair-count', dtype=float), "atompair-count"),
                (MoleculeTransformer(featurizer='avalon', dtype=float), "avalon"),
                (MoleculeTransformer(featurizer='ecfp-count', dtype=float), "ecfp-count"),
                (MoleculeTransformer(featurizer='ecfp', dtype=float), "ecfp"),
                (MoleculeTransformer(featurizer='erg', dtype=float), "erg"),
                (MoleculeTransformer(featurizer='estate', dtype=float), "estate"),
                (MoleculeTransformer(featurizer='fcfp-count', dtype=float), "fcfp-count"),
                (MoleculeTransformer(featurizer='fcfp', dtype=float), "fcfp"),
                (MoleculeTransformer(featurizer='maccs', dtype=float), "maccs"),
                (MoleculeTransformer(featurizer='pattern', dtype=float), "pattern"),
                (MoleculeTransformer(featurizer='rdkit', dtype=float), "rdkit"),
                (MoleculeTransformer(featurizer='topological-count', dtype=float), "topological-count"),
                (MoleculeTransformer(featurizer='topological', dtype=float), "topological"),
                
                #long
                (MoleculeTransformer(featurizer='desc2D', dtype=float), "desc2D"),
                (MoleculeTransformer(featurizer=Pharmacophore2D(factory='cats'), dtype=float), "pharm2D-cats"),
                (MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float), "pharm2D-gobbi"),
                (MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float), "pharm2D-pmapper"),
            ]

ml_list = [ 
            (Ridge(), "RidgeRegression"),
            (LinearRegression(), 'LinearRegression'),
            (RandomForestRegressor(), 'RandomForestRegressor'),
            (MLPRegressor(), 'MLPRegressor'),
            (SVR(), 'SVR'),
            (KNeighborsRegressor(), 'KNeighborsRegressor')
           ]

# input data
benchmark_collection =  Path("../benchmark_collection_original").resolve()

# output data and calculations
if __name__ == "__main__":
    prediction_collection = Path("benchmark_model_prediction").resolve()

    if prediction_collection.exists():
        shutil.rmtree(prediction_collection)

    with Pool(cpu_count() - 1) as pool:
            list(pool.starmap(partial(process, 
                                prediction_collection=prediction_collection, 
                                descr_list=descr_list, 
                                ml_list=ml_list),
                                    prepare(benchmark_collection)))