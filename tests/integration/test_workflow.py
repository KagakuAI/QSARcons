import pytest
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

# -----------------------------
# Regression Test
# -----------------------------
def test_consensus_search(regression_folder, consensus_searchers):
    df_val = pd.read_csv(f"{regression_folder}/val.csv")
    df_test = pd.read_csv(f"{regression_folder}/test.csv")

    x_val, y_val = df_val.iloc[:, 2:], df_val.iloc[:, 1]
    x_test, y_test = df_test.iloc[:, 2:], df_test.iloc[:, 1]

    for name, searcher in consensus_searchers:
        best = searcher.run(x_val, y_val)
        pred_val = searcher.predict_cons(x_val[best])
        pred_test = searcher.predict_cons(x_test[best])

        # sanity checks
        assert len(pred_val) == len(y_val)
        assert len(pred_test) == len(y_test)

        # optionally print R² for quick inspection
        print(f"[Regression] {name}: R2 val={r2_score(y_val, pred_val):.3f}, test={r2_score(y_test, pred_test):.3f}")

# -----------------------------
# Classification Test
# -----------------------------
def test_classification_consensus(classification_folder, consensus_searchers):
    df_val = pd.read_csv(f"{classification_folder}/val.csv")
    df_test = pd.read_csv(f"{classification_folder}/test.csv")

    x_val, y_val = df_val.iloc[:, 2:], df_val.iloc[:, 1]
    x_test, y_test = df_test.iloc[:, 2:], df_test.iloc[:, 1]

    for name, searcher in consensus_searchers:
        best = searcher.run(x_val, y_val)
        pred_val = searcher.predict_cons(x_val[best])
        pred_test = searcher.predict_cons(x_test[best])

        # sanity checks
        assert len(pred_val) == len(y_val)
        assert len(pred_test) == len(y_test)
        assert set(pred_val).issubset(set(y_val.unique()))
        assert set(pred_test).issubset(set(y_test.unique()))

        # optionally print balanced accuracy
        print(f"[Classification] {name}: val={roc_auc_score(y_val, pred_val):.3f}, test={roc_auc_score(y_test, pred_test):.3f}")
