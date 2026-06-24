
QSARcons - smart search for consensus of QSAR models
--------------------------------------------------------------------

The motivation behind this project is that there are many available chemical descriptors and machine learning methods,
and usually, it is not obvious which combination to prefer for modelling the target property of the molecule.
Therefore, the idea is just to build multiple (>100) simple individual QSAR models with diverse descriptors and algorithms,
and then a genetic algorithm to smartly optimize the subset of models delivering the best performance on the validation dataset.

Motivation
--------------------------------------------------------------------

**1. Simple design**: ``QSARcons`` focuses on simplicity of use. The default pipeline just requires training and test data.

**2. Traditional QSAR**: ``QSARcons`` includes a wide range of traditional molecular descriptors and machine learning algorithms, providing a transparent baseline for comparison with more advanced approaches like deep learning-based or complex QSAR workflows.

**3. Universal workflow** - ``QSARcons`` can be applied to any type of chemical property modelling.

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

    from datasets import load_dataset
    from qsarcons.meta import ConsensusModel

    train_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="train").to_pandas()
    test_df = load_dataset("openadmet/openadmet-expansionrx-challenge-data", split="test").to_pandas()

    prop_name = "Caco-2 Permeability Efflux"
    train_df = train_df[["SMILES", prop_name]].dropna()
    test_df = test_df[["SMILES", prop_name]].dropna()

    output_folder = f"{prop_name}_qsarcons"

    model = ConsensusModel(hopt=False, output_folder=output_folder, verbose=True)
    test_df_pred = model.run_predict(train_df, test_df)

    print(model.best_cons)

Colab
---------------------------------------------------------------------

See an example in `QSARcons pipeline <https://colab.research.google.com/github/KagakuAI/QSARcons/blob/main/colab/Notebook_1_QSARcons_pipeline.ipynb>`_ .

QSARcons Basic vs. QSARcons Pro
---------------------------------------------------------------------
The QSARcons idea is that diverse and strong individual models can be combined to even stronger consensus.
Currently, two versions are under development:

- **QSARcons Basic:** includes ``RDKit`` descriptors + ``scikit-learn`` ML methods
- **QSARcons Pro:** will include **QSARcons Basic** + other workflows for building individual models (e.g. ``chemprop`` and ``QSARmil``) to combine traditional and advanced modelling approaches into stronger consensuses
