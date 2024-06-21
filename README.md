# Cifar_MLRF

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projet de Machine Learning et Reconnaissance de Formes.

Ce projet permet la reconnaissance et la classification d'images, de la base de données Cifar, en 10 classes différentes, et ce grâce à des méthodes de machine learning.

## Project Organization

```
├── LICENSE            <- Open-source MIT license
├── Makefile           <- Makefile with convenience commands like `make install` or `make run`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   └── ...
│
├── notebooks          <- Jupyter notebooks.
│   └── verif_steps.ipynb           <- Jupyter notebook to get data, train, test and get figures about the results
│   └── verif_features.ipynb        <- Jupyter notebook to test features
│   └── decision_boundaries.ipynb   <- Jupyter notebook to get decision boudaries plots                   
│
├── pyproject.toml     <- Project configuration file with package metadata for MLRF
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│   │   └── ...
│   └── ...
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── MLRF               <- Source code for use in this project
    │
    ├── __init__.py    <- Makes MLRF a Python module
    │
    ├── config.py      <- Scripts to configurate paths etc.
    │
    ├── dataset.py     <- Scripts to download and generate data
    │
    ├── features.py    <- Scripts to turn raw data into features for modeling
    │
    ├── model_utils.py <- Scripts to define utilities like pipeline, classifiers, and load/save methods
    │
    ├── modeling       <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict.py
    │   └── train.py
    │
    └── plots.py       <- Scripts to create exploratory and results oriented visualizations
    │
    └── setup.py       <- Scipt to create a package from the project
```

--------

## The CIFAR-10 dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. They were collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.