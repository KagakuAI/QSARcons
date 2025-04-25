import os
import sys
import random
import pandas as pd
from pathlib import Path
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from qsarcons.consensus import RandomSearchRegressor, SystematicSearchRegressor, GeneticSearchRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge

from multiprocessing import Pool, cpu_count
from functools import partial


def prepare_dirs(prediction_collection):
    output = []
    for coll_folder in os.listdir(prediction_collection):
        for bench_name in os.listdir(os.path.join(prediction_collection, coll_folder)):
            output.append((prediction_collection, coll_folder, bench_name))
    return output


def process(estimators, method_list, prediction_consensus, prediction_collection, bench_folder, bench_name):
    # load data
    df_test = pd.read_csv(os.path.join(prediction_collection, bench_folder, bench_name, f"combinedtest.csv"))
    df_train = pd.read_csv(os.path.join(prediction_collection, bench_folder, bench_name, f"combinedtrain.csv"))
    df_test_default = pd.read_csv(os.path.join(prediction_collection, bench_folder, bench_name, f"combinedtest_default.csv"))
    df_train_default = pd.read_csv(os.path.join(prediction_collection, bench_folder, bench_name, f"combinedtrain_default.csv"))

    res_folder = os.path.join(prediction_consensus, bench_folder, bench_name)
    os.makedirs(res_folder, exist_ok=True)

    # remove y_true column prof predictions table
    x_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]
    x_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
    x_test_default  = df_test_default.iloc[:, 1:]
    x_train_default = df_train_default.iloc[:, 1:]

    # save y_true columns
    res_df_test_default, res_df_test = pd.DataFrame({'Y_TRUE': y_test}), pd.DataFrame({'Y_TRUE': y_test})
    res_df_train_default, res_df_train = pd.DataFrame({'Y_TRUE': y_train}), pd.DataFrame({'Y_TRUE': y_train})

    # build consensus
    for method_func, method_name in method_list:

        # find consensus training set
        cons = method_func.run(x_train_default, y_train)
        y_pred = x_train_default[cons].mean(axis=1)
        res_df_train_default[method_name] = y_pred

        # testset
        y_pred = x_test_default[cons].mean(axis=1)
        res_df_test_default[method_name] = y_pred

        # find consensus training set hyperopt
        cons = method_func.run(x_train, y_train)
        y_pred = x_train[cons].mean(axis=1)
        res_df_train[method_name] = y_pred

        # testset with hyperopt
        y_pred = x_test[cons].mean(axis=1)
        res_df_test[method_name] = y_pred

    #stacking
    for model, model_name in estimators:
        model.fit(x_train_default, y_train)

        # Train predictions
        train_pred = model.predict(x_train_default)
        res_df_train_default[f"Stacking{model_name}"] = train_pred

        # Test predictions
        test_pred = model.predict(x_test_default)
        res_df_test_default[f"Stacking{model_name}"] = test_pred

        # Hyperoptimized models
        model.fit(x_train, y_train)

        # Train predictions 
        train_pred = model.predict(x_train)  
        res_df_train[f"Stacking{model_name}"] = train_pred

        # Test predictions
        test_pred = model.predict(x_test)
        res_df_test[f"Stacking{model_name}"] = test_pred

    res_df_test_default.to_csv(os.path.join(prediction_consensus, bench_folder, bench_name, f"default_test.csv"), index = False)
    res_df_train_default.to_csv(os.path.join(prediction_consensus, bench_folder, bench_name, f"default_train.csv"), index = False)
    res_df_test.to_csv(os.path.join(prediction_consensus, bench_folder, bench_name, f"test.csv"), index = False)
    res_df_train.to_csv(os.path.join(prediction_consensus, bench_folder, bench_name, f"train.csv"), index = False)

random.seed(42)


method_list = [
               (SystematicSearchRegressor(cons_size=10, metric='r2'), 'Best'),
               (SystematicSearchRegressor(cons_size=10**3, metric='r2'), 'All'),
               (RandomSearchRegressor(cons_size=2, n_iter=5, metric='r2'), 'Random'),
               (SystematicSearchRegressor(cons_size=2, metric='r2'), 'Systematic'),
               (GeneticSearchRegressor(cons_size=3, mut_prob=0.5, metric='r2'), 'Genetic'),
               #(HyperoptSearchRegressor(cons_size=3, n_iter=200, metric='rmse'), 'Hyperopt')
               ]

estimators = [
            (LinearRegression(), 'LinearRegression'),
            (RandomForestRegressor(), 'RandomForestRegressor'),
            (MLPRegressor(), 'MLPRegressor'),
            (Ridge(), 'Ridge'),
            (SVR(), 'SVR'),
            (KNeighborsRegressor(), 'KNeighborsRegressor')
            ]

prediction_collection =  Path("../benchmark_model_prediction").resolve()
prediction_consensus =  Path("../benchmark_model_consensus").resolve()
os.makedirs(prediction_consensus, exist_ok=True)

if __name__ == "__main__":
    if prediction_consensus.exists():
        shutil.rmtree(prediction_consensus)

    with Pool(cpu_count() -1) as pool:
        list(pool.starmap(partial(process,
                                    estimators,
                                    method_list, 
                                    prediction_consensus),
                                    prepare_dirs(prediction_collection)))

            
