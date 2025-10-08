# ==========================================================
# Imports
# ==========================================================
import os
import hashlib
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from filelock import FileLock

from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor  # You already use it
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeClassifier, BayesianRidge
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.utils.multiclass import type_of_target

from sklearn.preprocessing import MinMaxScaler
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

from .hopt import StepwiseHopt, DEFAULT_PARAM_GRID_REGRESSORS, DEFAULT_PARAM_GRID_CLASSIFIERS

from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# ==========================================================
# Configuration
# ==========================================================
DESCRIPTORS = {

    # fingerprints
    "avalon": MoleculeTransformer(featurizer='avalon', dtype=float),
    "rdkit": MoleculeTransformer(featurizer='rdkit', dtype=float),
    "maccs": MoleculeTransformer(featurizer='maccs', dtype=float),
    "atompair-count": MoleculeTransformer(featurizer='atompair-count', dtype=float),
    "fcfp": MoleculeTransformer(featurizer='fcfp', dtype=float),
    "fcfp-count": MoleculeTransformer(featurizer='fcfp-count', dtype=float),
    "ecfp": MoleculeTransformer(featurizer='ecfp', dtype=float),
    "ecfp-count": MoleculeTransformer(featurizer='ecfp-count', dtype=float),
    "topological": MoleculeTransformer(featurizer='topological', dtype=float),
    "topological-count": MoleculeTransformer(featurizer='topological-count', dtype=float),
    "secfp": MoleculeTransformer(featurizer='secfp', dtype=float),

    # scaffold
    "scaffoldkeys": MoleculeTransformer(featurizer='scaffoldkeys', dtype=float),

    # phys-chem
    "desc2D": MoleculeTransformer(featurizer='desc2D', dtype=float),

    # electrotopological
    "estate": MoleculeTransformer(featurizer='estate', dtype=float),

    # pharmacophore
    "erg": MoleculeTransformer(featurizer='erg', dtype=float),
    "cats2d": MoleculeTransformer(featurizer='cats2d', dtype=float),
    "pharm2D-cats": MoleculeTransformer(featurizer=Pharmacophore2D(factory='cats'), dtype=float),
    "pharm2D-gobbi": MoleculeTransformer(featurizer=Pharmacophore2D(factory='gobbi'), dtype=float),
    "pharm2D-pmapper": MoleculeTransformer(featurizer=Pharmacophore2D(factory='pmapper'), dtype=float),
}

REGRESSORS = {
    "RidgeRegression": Ridge,
    "PLSRegression": PLSRegression,
    "KNeighborsRegressor": KNeighborsRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "MLPRegressor": MLPRegressor,
    "SVR": SVR,
}

CLASSIFIERS = {
    "RidgeClassifier": RidgeClassifier,
    "LogisticRegression": LogisticRegression,  # Closest to ElasticNet / Lasso in classification
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "MLPClassifier": MLPClassifier,
    "SVC": SVC,
}

# ==========================================================
# Utility Functions
# ==========================================================
def write_model_predictions(model_name, smiles_list, y_true, y_pred, output_path):
    """Append new model predictions as a column to CSV assuming fixed row order."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lockfile = os.path.join(tempfile.gettempdir(), f"{hashlib.md5(output_path.encode()).hexdigest()}.lock")

    new_col = pd.DataFrame({model_name: y_pred})

    with FileLock(lockfile):
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df[model_name] = new_col[model_name]
        else:
            df = pd.DataFrame({
                "SMILES": smiles_list,
                "Y_TRUE": y_true,
                model_name: y_pred
            })

        # Optional: reorder columns for readability
        cols = ["SMILES", "Y_TRUE"] + sorted(c for c in df.columns if c not in {"SMILES", "Y_TRUE"})
        df = df[cols]

        df.to_csv(output_path, index=False)

def replace_nan_with_column_mean(x):
    # Convert None to np.nan if present
    x = np.array(x, dtype=float)

    # Calculate column means ignoring NaNs
    col_means = np.nanmean(x, axis=0)

    # Find indices where NaN values are located
    inds = np.where(np.isnan(x))

    # Replace NaNs with respective column means
    x[inds] = np.take(col_means, inds[1])

    return x

# ==========================================================
# ModelBuilder Class
# ==========================================================
class BasicBuilder:
    def __init__(self, descriptor, estimator, hopt, model_name, model_folder):
        self.descriptor = descriptor
        self.estimator = estimator
        self.hopt = hopt
        self.model_name = model_name
        self.model_folder = model_folder

    def calc_descriptors(self, df_data):
        """Load SMILES and properties from CSV."""
        smi, y = df_data.iloc[:, 0], df_data.iloc[:, 1]
        x = self.descriptor(smi)
        x = replace_nan_with_column_mean(x)
        return smi, x, y

    def scale_descriptors(self, x_train, x_val, x_test):
        """Scale molecular descriptors using MinMaxScaler."""
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    def run(self, df_train, df_val, df_test):
        """Full training + prediction pipeline with optional stepwise hyperopt."""

        # 1. Calculate descriptors
        smi_train, x_train, y_train = self.calc_descriptors(df_train)
        smi_val, x_val, y_val = self.calc_descriptors(df_val)
        smi_test, x_test, y_test = self.calc_descriptors(df_test)

        # 2. Scale descriptors
        x_train_scaled, x_val_scaled, x_test_scaled = self.scale_descriptors(
            x_train, x_val, x_test
        )

        # 3. Train estimator (optionally with stepwise hyperopt)
        if self.hopt:
            est_name = self.estimator.__name__
            # Detect regression vs classification
            task_type = type_of_target(y_train)
            is_classification = task_type in ["binary", "multiclass"]

            # Choose appropriate parameter grid
            if is_classification:
                param_grid = DEFAULT_PARAM_GRID_CLASSIFIERS.get(est_name)
                scoring = "accuracy"
            else:
                param_grid = DEFAULT_PARAM_GRID_REGRESSORS.get(est_name)
                scoring = "r2"

            if param_grid is None:
                raise ValueError(f"No default parameter grid defined for {est_name}.")

            # Perform stepwise optimization
            opt = StepwiseHopt(self.estimator(), param_grid, scoring=scoring, cv=3, verbose=False)
            opt.fit(x_train_scaled, y_train)
            estimator = opt.estimator
            estimator.fit(x_train_scaled, y_train)
        else:
            estimator = self.estimator()
            estimator.fit(x_train_scaled, y_train)

        # 4. Make validation/test predictions
        pred_val = list(estimator.predict(x_val_scaled))
        pred_test = list(estimator.predict(x_test_scaled))

        # 5. Save predictions
        write_model_predictions(
            self.model_name, smi_val, y_val, pred_val, os.path.join(self.model_folder, "val.csv")
        )
        write_model_predictions(
            self.model_name, smi_test, y_test, pred_test, os.path.join(self.model_folder, "test.csv")
        )

        return self

class LazyML:
    def __init__(self, task="regression", hopt=False, output_folder=None, verbose=True):
        self.task = task
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

    def run(self, df_train, df_val, df_test):
        all_models = []
        for desc_name, descriptor in DESCRIPTORS.items():
            for est_name, estimator in REGRESSORS.items():
                model_name = f"BasicML|{desc_name}|{est_name}"
                os.makedirs(self.output_folder, exist_ok=True)

                model = BasicBuilder(
                    descriptor=descriptor,
                    estimator=estimator,
                    hopt=self.hopt,
                    model_name=model_name,
                    model_folder=self.output_folder,
                )
                all_models.append(model)

        results = []
        with tqdm(total=len(all_models), disable=not self.verbose) as pbar:
            for model in all_models:
                model.run(df_train, df_val, df_test)
                results.append(model.model_name)
                pbar.update(1)

        return None








