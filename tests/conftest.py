import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from qsarcons.lazy import LazyML
from qsarcons.consensus import RandomSearch, SystematicSearch, GeneticSearch

# -----------------------------
# Dataset loader
# -----------------------------
def load_data():
    url = (
        "https://huggingface.co/datasets/KagakuData/Notebooks/"
        "resolve/main/chembl_200/CHEMBL1821.csv"
    )
    df = pd.read_csv(url, header=None)[[0, 2]].iloc[:140]
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    return df_train, df_val, df_test

# -----------------------------
# Regression + Classification Data
# -----------------------------
@pytest.fixture
def regression_data():
    return load_data()

@pytest.fixture
def classification_data():
    df_train, df_val, df_test = load_data()
    for df in (df_train, df_val, df_test):
        df.iloc[:, 1] = (df.iloc[:, 1] > 6.5).astype(int)
    return df_train, df_val, df_test

# -----------------------------
# LazyML Training
# -----------------------------
@pytest.fixture
def regression_folder(regression_data):
    df_train, df_val, df_test = regression_data
    out = "regression_models"
    lazy = LazyML(task="regression", hopt=True, output_folder=out, verbose=False)
    lazy.run(df_train, df_val, df_test)
    return out

@pytest.fixture
def classification_folder(classification_data):
    df_train, df_val, df_test = classification_data
    out = "classification_models"
    lazy = LazyML(task="classification", hopt=True, output_folder=out, verbose=False)
    lazy.run(df_train, df_val, df_test)
    return out

@pytest.fixture
def consensus_searchers():
    metric = "auto"
    cons_size = "auto"
    return [
        ("Best", SystematicSearch(cons_size=1, metric=metric)),
        ("Random", RandomSearch(cons_size=cons_size, n_iter=50, metric=metric)),
        ("Systematic", SystematicSearch(cons_size=cons_size, metric=metric)),
        ("Genetic", GeneticSearch(cons_size=cons_size, n_iter=20, metric=metric))
    ]

