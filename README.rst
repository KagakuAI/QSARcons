
QSARcons - smart searching for consensus of QSAR models
--------------------------------------------------------------------

The motivation behind this project is that there any many available chemical descriptors and machine learning methods,
and usually it is not obvious which combination to prefer for modelling the target property of the molecule.
Therefore, the idea is just to build multiple (>100) simple individual QSAR models with diverse descriptors and algorithms,
and then genetic algorithm to smartly optimize the subset of models delivering the best performance on the validation dataset.

Motivation
--------------------------------------------------------------------

**1. Simple design**: ``QSARcons`` focuses on simplicity of use. The default pipeline just requires training and test data.

**2. Traditional QSAR**: ``QSARcons`` includes a wide range of traditional molecular descriptors and machine learning algorithms, providing a transparent baseline for comparison with more advanced approaches like deep learning-based or complex QSAR workflows.

**3. Universal workflow** - ``QSARcons`` cab be applied to any type of chemical property modelling.

Overview
--------------------------------------------------------------------
QSARcons provides a two-layer workflow.

**1. Model generation**
   Build multiple QSAR models (>100) using 2D chemical descriptors and traditional machine learning algorithms.
   The individual model building pipeline is kept simple, without advanced data preprocessing. Optional in-house stepwise hyperparameter
   optimization is available for all ML methods.

**2. Consensus search**
   Identify the optimal subset of QSAR models using several search strategies:

- Random search
- Systematic search
- Genetic search

Installation
--------------------------------------------------------------------

.. code-block:: bash

    pip install qsarcons

QSARcons benchmarking
--------------------------------------------------------------------
``QSARcons`` can be easily benchmarked against alternative approaches. For that, just call the default pipeline function below.
Input data are dataframes where the first column is molecule SMILES and the second column is molecule property (regression or binary classification).

.. code-block:: python

    import polaris
    from sklearn.model_selection import train_test_split
    from qsarcons.cli import run_qsarcons

    # Load Polaris benchmark
    benchmark = polaris.load_benchmark("tdcommons/caco2-wang")
    data_train, data_test = benchmark.get_train_test_split()

    df_train, df_test = data_train.as_dataframe(), data_test.as_dataframe()
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

    # Run QSARcons
    test_pred = run_qsarcons(df_train, df_val, df_test, task="regression", output_folder="results")

    # Evaluate predictions
    results = benchmark.evaluate(test_pred)

Colab
---------------------------------------------------------------------

See an example in `QSARcons pipeline <https://colab.research.google.com/github/KagakuAI/QSARcons/blob/main/colab/Notebook_1_QSARcons_pipeline.ipynb>`_ .

QSARcons Basic vs. QSARcons Pro
---------------------------------------------------------------------

QSARcons idea is that diverse and strong individual models can be combined to even stronger consensus.
Currently, two versions are under development:

- **QSARcons Basic:** includes ``RDKit`` descriptors + ``scikit-learn`` ML methods
- **QSARcons Pro:** will include **QSARcons Basic** + other workflows for building individual models (e.g. ``chemprop`` and ``QSARmil``) to combine traditional and advanced modelling approaches into stronger consensuses