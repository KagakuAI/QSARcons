# QSARcons
QSAR consensus prediction

Install core dependencies:

Create and activate a new conda environment:

    conda create -n qsarcons python==3.9
    conda activate qsarcons

    conda install molfeat[all]
    conda install xgboost==2.1.4


Optional dependencies for specific descriptors:
    
    # map descriptors
    conda install tmap::tmap
    pip install git+https://github.com/reymond-group/map4@v1.0

    # pretrained models
    conda install transformers tokenizers selfies 
    conda install -c conda-forge dgllife
    pip install torch==2.1.0 torchdata==0.7.0 dgl==1.1.0 
