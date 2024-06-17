from learn_diagonal_gauss import bayesGMM
from learn_diagonal_gauss_fs import bayesGMM_FS
from learn_cat_gauss_diag import bayesCGMM
from learn_diagonal_cat import catMM
import utils
import numpy as np

# Class for Gaussian features with a diagonal covariance matrix
class DiagGaussian(object):
    # Initialize hyperparameters
    def __init__(self, alpha: float, m_0_partial: float, k_0: int, v_0_partial: int, S_0_partial: float, K_initial=40, iterations=40, runs=1, seed=732843):

        """
        alpha: Dirichlet hyperparameter for mixing probabilities
        m_0_partial: Partial value of m_0, where m_0 is prior mean for the Gaussian data (m_0)
        k_0: How strong we are about the above prior, m0 
        S_0_partial: Proportional to prior mean for variance
        v_0_partial: How strong we are about the above prior, S_0
        K_initial: Initial total number of clusters
        iterations: Total number of iterations
        runs: Total number of training runs
        seed: Random seed
        
        The hyperparameters gets updated as: 
    
        m_N = (k_0 * m0 + mean(x)) / k_N
        k_N = k_0 + N
        S_N = S_0 + N * s^2 + k_0 * m_0^2 - k_N * m_N^2 + sum (x_i)^2
        v_N = v_0 + N

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

    # Sample data using the Gibbs sampling method
    def sample(self, data: float, printIterResults=False):

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

    # Return the cluster assignments
    def assignments(self):
        return self.model.z_map


# Class for categorical features
class Categorical(object):
    # Initialize hyperparameters
    def __init__(self, alpha: float, gamma: float, K_initial = 40, iterations = 40, runs = 1, seed = 39284):

        """
        alpha: Dirichlet hyperparameter for data point assignment mixing probabilities
        gamma: Dirichlet hyperparameter for Categories mixing probabilities
        """

        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed

    # Sample data using the Gibbs sampling method
    def sample(self, C, printIterResults=False):
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

    # Return the cluster assignments
    def assignments(self):
        return self.model.z_map


# Class for Gaussian features with a diagonal covariance matrix and feature selection
class DiagGaussianFS(object):
    # Initialize hyperparameters
    def __init__(self, alpha: float, m_0_partial: float, k_0: float, v_0_partial: int, S_0_partial: float, FS = False, K_initial = 40, iterations = 40, runs = 1, seed = 238627):

        """
        alpha: Dirichlet hyperparameter for mixing probabilities
        m_0_partial: Partial value of m_0, where m_0 is prior mean for the Gaussian data (m_0)
        k_0: How strong we are about the above prior, m0 
        S_0_partial: Proportional to prior mean for variance
        v_0_partial: How strong we are about the above prior, S_0
        FS: True if want to incorporate feature selection
        K_initial: Initial total number of clusters
        iterations: Total number of iterations
        runs: Total number of training runs
        seed: Random seed
        """

        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.FS = FS
        self.seed = seed

    # Sample data using the Gibbs sampling method
    def sample(self, data: float, printIterResults=False):
        
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

    # Return the cluster assignments
    def assignments(self):
        return self.model.z_map


# Class for a combined model with Gaussian and categorical features
class DiagGaussianCategorical(object):
    # Initialize hyperparameters
    def __init__(self, alpha: float, m_0_partial: float, k_0: float, v_0_partial: int, S_0_partial: float, gamma: float, K_initial=40, iterations=40, runs=1, seed=39284):

        """
        alpha: Dirichlet hyperparameter for mixing probabilities
        m_0_partial: Partial value of m_0, where m_0 is prior mean for the Gaussian data (m_0)
        k_0: How strong we are about the above prior, m0 
        S_0_partial: Proportional to prior mean for variance
        v_0_partial: How strong we are about the above prior, S_0
        gamma: Dirichlet hyperparameter for Categories mixing probabilities
        K_initial: Initial total number of clusters
        iterations: Total number of iterations
        runs: Total number of training runs
        seed: Random seed
        """
        
        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed

    # Sample data using the Gibbs sampling method
    def sample(self, X, C, printIterResults=False):
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

    # Return the cluster assignments
    def assignments(self):
        return self.model.z_map