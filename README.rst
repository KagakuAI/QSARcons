QSARcons – a tool for searching optimal consensus of QSAR models
-----------------------------

``QSARcons`` is a package for finding the optimal consensus of QSAR models for molecular property prediction. 
This solution is motivated by the fact, that there are many different chemical descriptors and machine-learning methods 
which can be combined into multiple possible QSAR models. So, the smart selection of the optimal subset of QSAR models 
(consensus) can be reasonable for many applications. However, consensus optimizers in ``QSARcons`` are not limited to 
QSAR applications, but to other ML problems where there is a need to find the optimal consensus of many ML models.

Installation
-----------------------------

``QSARcons`` can be installed with conda:

.. code-block:: bash
    
    # QSARcons
    conda create -n qsarcons python==3.9
    conda activate qsarcons
    
    # descriptors and ml methods for QSAR model building
    conda install molfeat[all]
    conda install xgboost==2.1.4

    # additional descriptors
    conda install tmap::tmap
    pip install git+https://github.com/reymond-group/map4@v1.0


Overview
-----------------------------

``QSARcons`` can be used for:

- Random consensus search
- Systematic consensus search
- Genetic consensus search


Usage
-----------------------------

Usage pipeline:

1. Load your dataset
2. Build multiple QSAR models
3. Use different consensus search methods from ``QSARcons``

See an example in `tutorial <tutorials/Tutorial_1_QSAR_consensus.ipynb>`_ .
    
    
    


