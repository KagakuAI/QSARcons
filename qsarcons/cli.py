import pandas as pd
from qsarcons.lazy import LazyML
from qsarcons.consensus import GeneticSearch


def run_qsarcons(df_train, df_val, df_test, task="regression", output_folder=None):

    # 1. Fill fake test prop
    if len(df_test.columns) == 1:
        df_test[1] = [None for i in df_test.index]

    # 2. Build multiple models
    lazy_ml = LazyML(task=task, hopt=True, output_folder=output_folder, verbose=True)
    lazy_ml.run(df_train, df_val, df_test)

    # 3. Load model predictions
    res_val = pd.read_csv(f"{output_folder}/val.csv")
    res_test = pd.read_csv(f"{output_folder}/test.csv")

    x_val, true_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]
    x_test = res_test.iloc[:, 2:]

    # 3. Run genetic search
    gen_search = GeneticSearch(cons_size="auto", metric="auto", n_iter=50)
    best_cons = gen_search.run(x_val, true_val)
    pred_test = gen_search.predict_cons(x_test[best_cons])

    print(f"Genetic consensus: {best_cons}")

    return pred_test