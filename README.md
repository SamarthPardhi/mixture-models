# Mixture Models

This repository contains implementations of my master's thesis work on various mixture models for heterogeneous data with feature selection. The models include Gaussian Mixture Models (GMM), Categorical Mixture Models, and combinations of both, with options for feature selection.

## Models

The `examples` directory contains Jupyter notebooks demonstrating the usage of following models on various datasets.

- `learn_general_gauss.py`: Gaussian mixture model with general covariance matrices
- `learn_diagonal_gauss.py`: Gaussian mixture model with diagonal covariance matrices
- `learn_diagonal_gauss_fs.py`: Gaussian Mixture Model (diagonal covariance matrix) with feature selection
- `learn_cat.py`: Categorical Mixture Model
- `learn_cat_fs.py`: Categorical Mixture Model with feature selection
- `learn_cat_gauss_diag.py`: Combined Gaussian and categorical mixture model

### Other Files

- `mixtureModels.py`: Main file containing the mixture model classes
- `learn_scikit_gmm.py`: Wrapper for scikit-learn's Gaussian Mixture Model
- `cluster_stats_new.py`: Helper functions for cluster statistics containing classes of each models
- `utils.py`: Utility functions for data handling and visualization

## Usage

The `mixtureModels.py` file provides a convenient way for interacting with the various mixture models implemented in this project.

Each model can be run from the command line with various arguments. For example:

Common arguments include:

- `-h`: Help to all the arguments
- `-f`: Input data file
- `-k`: Number of clusters (or maximum number if unknown)
- `-i`: Number of iterations
- `-o`: Output directory
- `-r`: Number of training runs
- `-t`: File with true labels (if available)

## Dependencies

- NumPy
- SciPy
- Matplotlib
- scikit-learn
- seaborn

## Contact

If you have any issues, questions, suggestions, or feedback regarding this repository, feel free to reach out to me: [samarthpardhi307@gmail.com].
