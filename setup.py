from setuptools import setup, find_packages

setup(
    name="qsarcons",
    version="1.0",
    author="KagakuAI",
    author_email="dvzankov@gmail.com.com",
    description="Consensus QSAR",
    long_description_content_type="text/x-rst",
    url="https://github.com/KagakuAI/QSARcons",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "torch_optimizer",
        "xgboost",
        "rdkit",
        "molfeat",
        "tqdm"

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # update if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
