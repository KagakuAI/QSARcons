QSARcons – a tool for searching optimal consensus of QSAR models
-----------------------------

``QSARcons`` is a package for finding the optimal consensus of QSAR models.
This project is motivated by the fact that there are many different chemical descriptors and machine learning methods
which can be combined into multiple QSAR models. So, a smart selection of the optimal subset of QSAR models (consensus)
can be reasonable for many applications.

Overview
-----------------------------

``QSARcons`` can be used for:

- Random consensus search
- Systematic consensus search
- Genetic consensus search

Installation
-----------------------------

``QSARcons`` can be installed with conda:

.. code-block:: bash
    
    # QSARcons
    conda create -n qsarcons python==3.9
    conda activate qsarcons


Tutorial
-----------------------------

See an example in `tutorial <tutorials/QSARcons_tutorial.ipynb>`_ .

Development
-----------------------------

``QSARcons`` is supposed to be used in two modes:

**1. Consensus optimizer.** In this case, the QSAR models are built by the user and their predictions of the validation (optimization)
set are submitted to ``QSARcons`` to find an optimal consensus. See the `tutorial <tutorials/QSARcons_tutorial.ipynb>`_ .

**2. Consensus QSAR builder.** In this case, the QSAR models are built automatically within the QSARcons pipeline and then
their best consensus is found. This is an end-to-end solution, where the user needs to submit only the training set.
This model now is under development and we are looking for robust pipelines for building multiple QSAR models to be integrated
into ``QSARcons``.

    
    


