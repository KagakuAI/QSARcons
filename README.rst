
QSARcons - smart searching for consensus of QSAR models
--------------------------------------------------------------------

``QSARcons`` is a package designed to identify optimal consensus combinations of QSAR models. The project is motivated
by the large number of available chemical descriptors and machine learning methods, which can be combined into many
different QSAR models. Selecting the most effective subset - and combining them into a consensus - can significantly
improve prediction accuracy and robustness.

Overview
--------------------------------------------------------------------
QSARcons provides a two-layer workflow.

**1. Model generation**
   Build multiple QSAR models (>100) using 2D chemical descriptors and traditional machine learning algorithms.
   The individual model building pipeline is kept simple, without advanced data preprocessing. Optional in-house stepwise hyperparameter
   optimization is available for all ML methods.

**2. Consensus optimization**
   Identify the optimal subset of QSAR models using several search strategies:

- Random search
- Systematic search
- Genetic search

Installation
--------------------------------------------------------------------

.. code-block:: bash

    pip install qsarcons

QSARcons benchmarking
------------------------
``QSARcons`` can be easily benchmarked against alternative approaches. For that, just call the default pipeline function below.
Input data are dataframes where the first column is molecule SMILES and the second column is molecule property
(regression or binary classification).

.. code-block:: python

    from qsarcons.cli import run_qsarcons

    test_pred = run_qsarcons(df_train, df_val, df_test, task="regression", output_folder="results")

Colab
---------------------------------------------------------------------

See an example in `QSARcons pipeline <https://colab.research.google.com/github/KagakuAI/QSARcons/blob/main/colab/Notebook_1_QSARcons_pipeline.ipynb>`_

