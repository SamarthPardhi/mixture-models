from learn_diagonal_gauss import bayesGMM
from learn_diagonal_gauss_fs import bayesGMM_FS
from learn_cat_gauss_diag import bayesCGMM
from learn_diagonal_cat import catMM

import utils
import numpy as np

# Gaussian Features with diagonal covariance matrix
class DiagGaussian(object):

    # Initialising the hyper-parameters
    def __init__(self, alpha, m_0_partial, k_0, v_0_partial, S_0_partial, K_initial=40, iterations=40, runs=1, seed = 732843):
        
        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed
        pass

    def sample(self, data, printIterResults=False):
        
        np.random.seed(self.seed)
        N = len(data)
        D = len(data[0])

        self.m_0, self.k_0, self.v_0, self.S_0 = self.m_0*np.ones(D), self.k_0, self.v_0 + D, self.S_0*(self.v_0+3)
        self.prior = utils.NIchi2(self.m_0, self.k_0, self.v_0, self.S_0)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run+1}")
            starting_assignments = []

            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)
        
            model = bayesGMM(data, self.prior, self.alpha, starting_assignments)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)
        
        least_BIC = 1*np.inf

        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC
                
        self.model = bestModel
        # self.modelResult

    def assignments(self):
        return self.model.z_map


class Categorical(object):

    def __init__(self, alpha, gamma, K_initial=40, iterations=40, runs=1, seed=39284):
    
        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed
        pass

    def sample(self, C, printIterResults=False):

        np.random.seed(self.seed)        
        N = len(C)
        cD = len(C[0])

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run+1}")
            starting_assignments = []

            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)
        
            model = catMM(C, self.alpha, self.gamma, starting_assignments)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)
        
        least_BIC = 1*np.inf

        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC
                
        self.model = bestModel


    def assignments(self):
        return self.model.z_map


class DiagGaussianFS(object):

    def __init__(self, alpha, m_0_partial, k_0, v_0_partial, S_0_partial, FS = False, K_initial=40, iterations=40, runs=1, seed = 238627):

        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.FS = FS
        self.seed = seed
        pass

    def sample(self, data, printIterResults=False):
        
        np.random.seed(self.seed)

        N = len(data)
        D = len(data[0])

        self.m_0, self.k_0, self.v_0, self.S_0 = self.m_0*np.ones(D), self.k_0, self.v_0 + D, self.S_0*(self.v_0+3)
        self.prior = utils.NIchi2(self.m_0, self.k_0, self.v_0, self.S_0)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run+1}")
            starting_assignments = []

            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)
        
            model = bayesGMM_FS(data, self.prior, self.alpha, starting_assignments, self.FS)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)
        
        least_BIC = 1*np.inf

        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC

        self.model = bestModel
        # self.modelResult

    def assignments(self):
        return self.model.z_map


class DiagGaussianCategorical(object):

    def __init__(self, alpha, m_0_partial, k_0, v_0_partial, S_0_partial, gamma, K_initial=40, iterations=40, runs=1, seed=39284):
    
        self.m_0, self.k_0, self.v_0, self.S_0 = m_0_partial, k_0, v_0_partial, S_0_partial
        self.alpha = alpha
        self.gamma = gamma
        self.K = K_initial
        self.iters = iterations
        self.runs = runs
        self.seed = seed
        pass

    def sample(self, X, C, printIterResults=False):

        np.random.seed(self.seed)        
        N = len(X)
        gD = len(X[0])
        cD = len(C[0])

        self.m_0, self.k_0, self.v_0, self.S_0 = self.m_0*np.ones(gD), self.k_0, self.v_0 + gD, self.S_0*(self.v_0+3)
        self.prior = utils.NIchi2(self.m_0, self.k_0, self.v_0, self.S_0)

        Models = []
        Results = []
        for run in range(self.runs):
            print(f"\nRun:  {run+1}")
            starting_assignments = []

            while len(set(starting_assignments)) != self.K:
                starting_assignments = np.random.randint(0, self.K, N)
        
            model = bayesCGMM(X, C, self.alpha, self.gamma, self.prior, starting_assignments)
            modelResult = model.gibbs_sampler(self.iters, run, toPrint=printIterResults, savePosterior=False)

            Models.append(model)
            Results.append(modelResult)
        
        least_BIC = 1*np.inf

        for model in Models:
            if model.BIC < least_BIC:
                bestModel = model
                least_BIC = model.BIC
                
        self.model = bestModel
        # self.modelResult

    def assignments(self):
        return self.model.z_map
    
