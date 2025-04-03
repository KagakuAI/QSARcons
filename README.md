# QSARcons
QSAR consensus prediction

Install Core Dependencies:

Create and activate a new Conda environment:

    conda create -n QSARcons python==3.9
    conda activate QSARcons

    conda install xgboost==2.1.4 molfeat[all]


Optional dependencies for specific descriptors:
    map4:
    conda install -c tmap:tmap
    pip install git+https://github.com/reymond-group/map4@v1.0

    #pretrained models
    conda install transformers tokenizers selfies dglteam 
    conda install -c conda-forge dgllife
    pip install torch==2.1.0 torchdata==0.7.0 dgl==1.1.0 
