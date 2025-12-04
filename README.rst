
QSARcons - smart searching for consensus of QSAR models
--------------------------------------------------------------------

``QSARcons`` is a package designed to identify optimal consensus combinations of QSAR models. The project is motivated
by the large number of available chemical descriptors and machine-learning methods, which can be combined into many
different QSAR models. Selecting the most effective subset — and combining them into a consensus — can significantly
improve prediction accuracy and robustness.

Overview
--------------------------------------------------------------------
QSARcons provides a two-layer workflow.

**1. Model generation**
   Build a large number of QSAR models (>100) using 2D chemical descriptors and traditional machine-learning algorithms.
   The modeling pipeline is intentionally simple, avoiding unnecessary preprocessing. Optional stepwise hyperparameter
   optimization is available for all ML methods.

**2. Consensus optimization**
   Identify the optimal subset of models using several search strategies:

- Random search
- Systematic search
- Genetic search

Installation
--------------------------------------------------------------------

.. code-block:: bash

    pip install qsarcons

Colab
---------------------------------------------------------------------

See an example in `QSARcons pipeline <https://colab.research.google.com/github/KagakuAI/QSARcons/blob/main/colab/Notebook_1_QSARcons_pipeline.ipynb>`_

