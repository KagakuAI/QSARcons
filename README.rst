
QSARcons - smart searching for consensus of QSAR models
--------------------------------------------------------------------

``QSARcons`` is a package designed to identify optimal consensus combinations of QSAR models. The project is built on
the idea that the vast number of available chemical descriptors and machine-learning algorithms can produce a large
variety of QSAR models. Selecting the most effective subset - and combining them into a consensus - can significantly
improve the accuracy of the final consensus predictions.

Overview
--------------------------------------------------------------------
``QSARcons`` has two main layers in the workflow.

1. Build multiple QSAR models (>100) using 2D chemical descriptors and traditional machine-learning algorithms.
The individual modeling pipeline is intentionally kept simple, avoiding complex preprocessing steps. Optional stepwise
hyperparameter optimization is available for all machine-learning methods.

2. Identify the optimal consensus of the generated QSAR models using multiple search strategies:

- Random search
- Systematic search
- Genetic search

Installation
--------------------------------------------------------------------

``QSARcons`` can be installed using conda/mamba package managers.

.. code-block:: bash

    pip install qsarcons

Tutorial
---------------------------------------------------------------------

See an example in `tutorial <notebooks/Notebook_1_LogS_pipeline.ipynb>`_ .
