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

from sklearn.preprocessing import MinMaxScaler
from molfeat.trans import MoleculeTransformer
from molfeat.calc.pharmacophore import Pharmacophore2D

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
    def __init__(self, descriptor, estimator, model_name, model_folder):
        self.descriptor = descriptor
        self.estimator = estimator
        self.model_name = model_name
        self.model_folder = model_folder

    def calc_descriptors(self, df_data):
        """Load SMILES and properties from CSV."""
        smi, y = df_data.iloc[:, 0], df_data.iloc[:, 1]
        x = self.descriptor(smi)
        x = replace_nan_with_column_mean(x)
        return smi, x, y

    def scale_descriptors(self, x_train, x_val, x_test):
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    def run(self, df_train, df_val, df_test):

        # 1. Calculate mol descriptors
        smi_train, x_train, y_train = self.calc_descriptors(df_train)
        smi_val, x_val, y_val = self.calc_descriptors(df_val)
        smi_test, x_test, y_test = self.calc_descriptors(df_test)

        # 2. Scale descriptors
        x_train_scaled, x_val_scaled, x_test_scaled = self.scale_descriptors(x_train, x_val, x_test)

        # 3. Train estimator
        estimator = self.estimator()
        estimator.fit(x_train_scaled, y_train)

        # 4. Make val/test predictions
        pred_val = list(estimator.predict(x_val_scaled))
        pred_test = list(estimator.predict(x_test_scaled))

        # 5. Save predictions
        write_model_predictions(self.model_name, smi_val, y_val, pred_val,
                                os.path.join(self.model_folder, "val.csv"))

        write_model_predictions(self.model_name, smi_test, y_test, pred_test,
                                os.path.join(self.model_folder, "test.csv"))

        return self

from joblib import Parallel, delayed
from tqdm import tqdm
import os, shutil
from joblib.parallel import BatchCompletionCallBack

class TqdmParallel(Parallel):
    """A Parallel subclass that updates tqdm progress bar in real time."""
    def __init__(self, tqdm_bar, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_bar = tqdm_bar

    def print_progress(self):
        # Disable joblib’s internal progress output
        pass

    def __call__(self, *args, **kwargs):
        with self.tqdm_bar:
            return super().__call__(*args, **kwargs)

    def _print(self, msg, *args, **kwargs):
        # Suppress joblib’s built-in printing
        pass

    def dispatch_one_batch(self, iterator):
        batch = super().dispatch_one_batch(iterator)
        if batch is not None:
            old_callback = batch[1]
            def new_callback(*args, **kwargs):
                self.tqdm_bar.update(1)
                return old_callback(*args, **kwargs)
            batch = (batch[0], new_callback)
        return batch

class LazyML:
    def __init__(self, task="regression", output_folder=None, verbose=True):
        self.task = task
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








