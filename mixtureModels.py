from learn_diagonal_gauss import bayesGMM
from learn_diagonal_gauss_fs import bayesGMM_FS
from learn_cat_gauss_diag import bayesCGMM
from learn_cat import catMM
from learn_cat_fs import catMM_FS
import utils
import numpy as np

# Gaussian mixture model with a diagonal covariance matrix
class DiagGaussian(object):
    def __init__(self, alpha: float, m_0_partial: float, k_0: int, v_0_partial: int, S_0_partial: float, K_initial=40, iterations=40, runs=1, seed=732843):
        """
        Initialize the DiagGaussian class with given hyperparameters.

        Parameters:
        alpha (float): Dirichlet hyperparameter for mixing probabilities
        m_0_partial (float): Partial value of m_0, the prior mean for the Gaussian data
        k_0 (int): Strength of the prior mean
        v_0_partial (int): Proportional to prior mean for variance
        S_0_partial (float): Strength of the prior variance
        K_initial (int): Initial total number of clusters
        iterations (int): Total number of iterations
        runs (int): Total number of training runs
        seed (int): Random seed for reproducibility

        We use Inverse-Chi-Squared which is a special case of Inverse-Gamma where:
        a = v_N / 2
        b = 1/2
        """
        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed

    def sample(self, data: np.ndarray, printIterResults=False):
        """
        This method initializes the model, performs multiple runs of Gibbs sampling,
        and selects the best model based on the Bayesian Information Criterion (BIC).
        
        Parameters:
        data (np.ndarray): Input data for sampling
        printIterResults (bool): Whether to print iteration results
        """
        np.random.seed(self.seed)
        N = len(data)
        D = len(data[0])

        # Update hyperparameters based on the data dimensions
        self.m_0, self.k_0, self.v_0, self.S_0 = self.m_0 * np.ones(D), self.k_0, self.v_0 + D, self.S_0 * (self.v_0 + 3)
        self.prior = utils.NIchi2(self.m_0, self.k_0, self.v_0, self.S_0)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run + 1}")
            starting_assignments = []

            # Ensure unique starting assignments
            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)

            # Initialize and run the Gaussian Mixture Model
            model = bayesGMM(data, self.prior, self.alpha, starting_assignments)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)

        # Select the best model based on BIC
        least_BIC = np.inf
        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC

        self.model = bestModel

    def assignments(self):
        """
        Get the cluster assignments.

        Returns:
        np.ndarray: Cluster assignments for each data point
        """
        return self.model.z_map

# Categorical mixture model with a diagonal covariance matrix
class Categorical(object):
    def __init__(self, alpha: float, gamma: float, K_initial=40, iterations=40, runs=1, seed=39284):
        """
        Initialize the Categorical class with given hyperparameters.

        Parameters:
        alpha (float): Dirichlet hyperparameter for data point assignment mixing probabilities
        gamma (float): Dirichlet hyperparameter for categories mixing probabilities
        K_initial (int): Initial total number of clusters
        iterations (int): Total number of iterations
        runs (int): Total number of training runs
        seed (int): Random seed for reproducibility
        """
        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed

    def sample(self, C: np.ndarray, printIterResults=False):
        """
        This method initializes the model, performs multiple runs of Gibbs sampling,
        and selects the best model based on the Bayesian Information Criterion (BIC).

        Parameters:
        C (np.ndarray): Input categorical data for sampling
        printIterResults (bool): Whether to print iteration results
        """
        np.random.seed(self.seed)
        N = len(C)
        cD = len(C[0])

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run + 1}")
            starting_assignments = []

            # Ensure unique starting assignments
            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)

            # Initialize and run the Categorical Mixture Model
            model = catMM(C, self.alpha, self.gamma, starting_assignments)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)

        # Select the best model based on BIC
        least_BIC = np.inf
        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC

        self.model = bestModel

    def assignments(self):
        """
        Get the cluster assignments.

        Returns:
        np.ndarray: Cluster assignments for each data point
        """
        return self.model.z_map

# Gaussian mixture model with a diagonal covariance matrix and feature selection incorporated
class DiagGaussianFS(object):
    def __init__(self, alpha: float, m_0_partial: float, k_0: float, v_0_partial: int, S_0_partial: float, FS=False, K_initial=40, iterations=40, runs=1, seed=238627):
        """
        Initialize the DiagGaussianFS class with given hyperparameters.

        Parameters:
        alpha (float): Dirichlet hyperparameter for mixing probabilities
        m_0_partial (float): Partial value of m_0, the prior mean for the Gaussian data
        k_0 (float): Strength of the prior mean
        v_0_partial (int): Proportional to prior mean for variance
        S_0_partial (float): Strength of the prior variance
        FS (bool): True if feature selection is to be incorporated
        K_initial (int): Initial total number of clusters
        iterations (int): Total number of iterations
        runs (int): Total number of training runs
        seed (int): Random seed for reproducibility
        """
        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.FS = FS
        self.seed = seed

    def sample(self, data: np.ndarray, printIterResults=False):
        """
        This method initializes the model, performs multiple runs of Gibbs sampling,
        and selects the best model based on the Bayesian Information Criterion (BIC).
        It incorporates feature selection if the FS parameter is set to True.

        Parameters:
        data (np.ndarray): Input data for sampling
        printIterResults (bool): Whether to print iteration results
        """
        np.random.seed(self.seed)
        N = len(data)
        D = len(data[0])

        # Update hyperparameters based on the data dimensions
        self.m_0, self.k_0, self.v_0, self.S_0 = self.m_0 * np.ones(D), self.k_0, self.v_0 + D, self.S_0 * (self.v_0 + 3)
        self.prior = utils.NIchi2(self.m_0, self.k_0, self.v_0, self.S_0)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run + 1}")
            starting_assignments = []

            # Ensure unique starting assignments
            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)

            # Initialize and run the Gaussian Mixture Model with Feature Selection
            model = bayesGMM_FS(data, self.prior, self.alpha, starting_assignments, self.FS)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)

        # Select the best model based on BIC
        least_BIC = np.inf
        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC

        self.model = bestModel

    def assignments(self):
        """
        Get the cluster assignments.

        Returns:
        np.ndarray: Cluster assignments for each data point
        """
        return self.model.z_map

# Gaussian and Categorical mixture model with a diagonal covariance matrix
class DiagGaussianCategorical(object):
    def __init__(self, alpha: float, m_0_partial: float, k_0: float, v_0_partial: int, S_0_partial: float, gamma: float, K_initial=40, iterations=40, runs=1, seed=39284):
        """
        Initialize the DiagGaussianCategorical class with given hyperparameters.

        Parameters:
        alpha (float): Dirichlet hyperparameter for mixing probabilities
        m_0_partial (float): Partial value of m_0, the prior mean for the Gaussian data
        k_0 (float): Strength of the prior mean
        v_0_partial (int): Proportional to prior mean for variance
        S_0_partial (float): Strength of the prior variance
        gamma (float): Dirichlet hyperparameter for categories mixing probabilities
        K_initial (int): Initial total number of clusters
        iterations (int): Total number of iterations
        runs (int): Total number of training runs
        seed (int): Random seed for reproducibility
        """
        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed

    def sample(self, X: np.ndarray, C: np.ndarray, printIterResults=False):
        """
        This method initializes the model, performs multiple runs of Gibbs sampling,
        and selects the best model based on the Bayesian Information Criterion (BIC).
        It combines both Gaussian and categorical data in the mixture model.
        
        Parameters:
        X (np.ndarray): Input Gaussian data for sampling
        C (np.ndarray): Input categorical data for sampling
        printIterResults (bool): Whether to print iteration results
        """
        np.random.seed(self.seed)
        N = len(X)
        gD = len(X[0])
        cD = len(C[0])

        # Update hyperparameters based on the Gaussian data dimensions
        self.m_0, self.k_0, self.v_0, self.S_0 = self.m_0 * np.ones(gD), self.k_0, self.v_0 + gD, self.S_0 * (self.v_0 + 3)
        self.prior = utils.NIchi2(self.m_0, self.k_0, self.v_0, self.S_0)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run + 1}")
            starting_assignments = []

            # Ensure unique starting assignments
            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)

            # Initialize and run the Combined Gaussian-Categorical Mixture Model
            model = bayesCGMM(X, C, self.alpha, self.gamma, self.prior, starting_assignments)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)

        # Select the best model based on BIC
        least_BIC = np.inf
        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC

        self.model = bestModel

    def assignments(self):
        """
        Get the cluster assignments.

        Returns:
        np.ndarray: Cluster assignments for each data point
        """
        return self.model.z_map

# Categorical mixture model with feature selection
class CategoricalFS(object):
    def __init__(self, alpha: float, gamma: float, FS=False, K_initial=40, iterations=40, runs=1, seed=39284):
        """
        Initialize the CategoricalFS class with given hyperparameters.

        Parameters:
        alpha (float): Dirichlet hyperparameter for data point assignment mixing probabilities
        gamma (float): Dirichlet hyperparameter for categories mixing probabilities
        FS (bool): True if feature selection is to be incorporated
        K_initial (int): Initial total number of clusters
        iterations (int): Total number of iterations
        runs (int): Total number of training runs
        seed (int): Random seed for reproducibility
        """
        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.FS = FS
        self.seed = seed

    def sample(self, C: np.ndarray, printIterResults=False):
        """
        This method initializes the model, performs multiple runs of Gibbs sampling,
        and selects the best model based on the Bayesian Information Criterion (BIC).
        It incorporates feature selection if the FS parameter is set to True.

        Parameters:
        C (np.ndarray): Input categorical data for sampling
        printIterResults (bool): Whether to print iteration results
        """
        np.random.seed(self.seed)
        N = len(C)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run + 1}")
            starting_assignments = []

            # Ensure unique starting assignments
            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)

            # Initialize and run the Categorical Mixture Model with Feature Selection
            model = catMM_FS(C, self.alpha, self.gamma, starting_assignments, self.FS)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)

        # Select the best model based on BIC
        least_BIC = np.inf
        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC

        self.model = bestModel

    def assignments(self):
        """
        Get the cluster assignments.

        Returns:
        np.ndarray: Cluster assignments for each data point
        """
        return self.model.z_map

