import numpy as np
from numpy.linalg import inv, slogdet
from scipy.special import gammaln
from scipy.stats import invwishart
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.stats import multivariate_t

# class for statistics of the Gaussian clusters with general covariance matrix
class gaussianClusters(object):
    """
    A class to represent and manage the statistics of Gaussian clusters with a general covariance matrix.
    
    Attributes:
        X (np.ndarray): Data points, a N x D array.
        prior: Prior parameters for the Gaussian clusters.
        alpha (float): Alpha parameter for the Dirichlet Process.
        K_max (int): Maximum number of clusters.
        assignments (np.ndarray): Initial assignments of clusters for each data point.
        sum_X (np.ndarray): Sum of data points for each cluster, a K x D array.
        outer_prod_X (np.ndarray): Outer product of data points for each cluster, a K x D x D array.
        counts (np.ndarray): Number of data points in each cluster, a K-dimensional vector.
        log_det_covariances (np.ndarray): Log determinant of covariance matrices for each cluster, a K-dimensional vector.
        inv_covariances (np.ndarray): Inverse of covariance matrices for each cluster, a K x D x D array.
        K (int): Current number of clusters.
    """

    def __init__(self, X: np.ndarray, prior, alpha: float, K: int, assignments: np.ndarray = None):
        """
        Initializes the GaussianClusters object with the given data and parameters.

        Args:
            X (np.ndarray): Data points, a N x D array.
            prior: Prior parameters for the Gaussian clusters.
            alpha (float): Alpha parameter for the Dirichlet Process.
            K (int): Maximum number of clusters.
            assignments (np.ndarray, optional): Initial assignments of clusters for each data point.
        """
        self.X = X
        self.N, self.D = X.shape
        self.prior = prior
        self.K_max = K
        self.alpha = alpha
        
        self.sum_X = np.zeros((self.K_max, self.D), float)
        self.outer_prod_X = np.zeros((self.K_max, self.D, self.D), float)
        self.counts = np.zeros(self.K_max, int)

        self.log_det_covariances = np.zeros(self.K_max)
        self.inv_covariances = np.zeros((self.K_max, self.D, self.D))

        self._cache()

        self.K = 0

        if assignments is None:
            self.assignments = -1 * np.ones(self.N, int)
        else:
            self.assignments = assignments
            for k in range(self.assignments.max() + 1):
                for i in np.where(self.assignments == k)[0]:
                    self.add_assignment(i, k)

    def _cache(self):
        """
        Caches various precomputed values to optimize computations.
        """
        self._cache_outer_X = np.zeros((self.N, self.D, self.D))
        for i in range(self.N):
            self._cache_outer_X[i, :, :] = np.outer(self.X[i], self.X[i])
        
        self._cache_prior_outer_m_0 = np.outer(self.prior.m_0, self.prior.m_0)

        Ns = np.concatenate([[1], np.arange(1, self.prior.v_0 + 2 * self.N + 4)])
        self._cache_gammaln_by_2 = gammaln(Ns / 2.0)
        self._cache_log_pi = np.log(np.pi)
        Ks = self.prior.k_0 + np.arange(0, 2 * self.N + 1)
        self._cache_log_Ks = np.log(Ks)
        self._cache_log_Vs = np.log(Ns)
        self._cache_gammaln_alpha = gammaln(self.alpha)
        self._cache_prod_k0m0 = self.prior.k_0 * self.prior.m_0
        self._cache_partial_S_sum = self.prior.S_0 + self.prior.k_0 * np.outer(self.prior.m_0, self.prior.m_0)
        covar_prior = (self.prior.k_0 + 1.0) / (self.prior.k_0 * (self.prior.v_0 - self.D + 1.0)) * self.prior.S_0
        self._cache_inv_covariance_prior = inv(covar_prior)
        self._cache_logdet_covariance_prior = slogdet(covar_prior)[1]
        self._cache_post_pred_prior_coeff = (
            self._cache_gammaln_by_2[self.prior.v_0 + 1] - self._cache_gammaln_by_2[self.prior.v_0 - self.D + 1] 
            - self.D / 2.0 * (self._cache_log_Vs[self.prior.v_0 - self.D + 1] + self._cache_log_pi) 
            - 0.5 * self._cache_logdet_covariance_prior
        )

    def cache_cluster_stats(self, k: int):
        """
        Caches the statistics for the k-th cluster.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Cached statistics (log_det_covariance, inv_covariance, count, sum_X, outer_prod_X).
        """
        return (
            self.log_det_covariances[k].copy(),
            self.inv_covariances[k].copy(),
            self.counts[k].copy(),
            self.sum_X[k].copy(),
            self.outer_prod_X[k].copy()
        )

    def restore_cluster_stats(self, k: int, log_det_covariance: float, inv_covariance: np.ndarray, count: int, sum_X: np.ndarray, outer_prod_X: np.ndarray):
        """
        Restores the cached statistics for the k-th cluster.

        Args:
            k (int): Index of the cluster.
            log_det_covariance (float): Log determinant of the covariance matrix.
            inv_covariance (np.ndarray): Inverse of the covariance matrix.
            count (int): Number of data points in the cluster.
            sum_X (np.ndarray): Sum of data points in the cluster.
            outer_prod_X (np.ndarray): Outer product of data points in the cluster.
        """
        self.log_det_covariances[k] = log_det_covariance
        self.inv_covariances[k, :, :] = inv_covariance
        self.counts[k] = count
        self.sum_X[k] = sum_X
        self.outer_prod_X[k] = outer_prod_X

    def add_assignment(self, i: int, k: int):
        """
        Assigns the i-th data point to the k-th cluster and updates the cluster statistics.

        Args:
            i (int): Index of the data point.
            k (int): Index of the cluster.
        """
        if k == self.K:
            self.K += 1
            self.sum_X[k, :] = np.zeros(self.D)
            self.outer_prod_X[k, :, :] = np.zeros((self.D, self.D))

        self.assignments[i] = k
        self.sum_X[k, :] += self.X[i]
        self.outer_prod_X[k, :, :] += self._cache_outer_X[i]
        self.counts[k] += 1
        self._update_log_det_covariance_and_inv_covariance(k)

    def del_assignment(self, i: int):
        """
        Deletes the assignment of the i-th data point and updates the cluster statistics.

        Args:
            i (int): Index of the data point.
        """
        k = self.assignments[i]

        if k != -1:
            self.assignments[i] = -1
            self.counts[k] -= 1
            if self.counts[k] == 0:
                self.empty_cluster(k)
            else:
                self.sum_X[k, :] -= self.X[i]
                self.outer_prod_X[k, :, :] -= self._cache_outer_X[i]
                self._update_log_det_covariance_and_inv_covariance(k)

    def empty_cluster(self, k: int):
        """
        Empties the k-th cluster by removing it and updating the remaining clusters.

        Args:
            k (int): Index of the cluster to be emptied.
        """
        self.K -= 1
        if k != self.K:
            self.sum_X[k, :] = self.sum_X[self.K, :]
            self.outer_prod_X[k, :, :] = self.outer_prod_X[self.K, :, :]
            self.counts[k] = self.counts[self.K]
            self.log_det_covariances[k] = self.log_det_covariances[self.K]
            self.inv_covariances[k, :, :] = self.inv_covariances[self.K, :, :]
            self.assignments[np.where(self.assignments == self.K)] = k

        self._update_log_det_covariance_and_inv_covariance_priors(self.K)
        self.counts[self.K] = 0
        self.sum_X[self.K, :] = np.zeros(self.D)
        self.outer_prod_X[self.K, :, :] = np.zeros((self.D, self.D))

    def log_post_pred_prior(self, i: int):
        """
        Computes the log posterior predictive probability of the i-th data point under the prior alone.

        Args:
            i (int): Index of the data point.

        Returns:
            float: Log posterior predictive probability.
        """
        return self._multivariate_students_t_prior(i)

    def log_post_pred(self, i: int):
        """
        Computes the log posterior predictive probabilities for the i-th data point.
        Equation 22 in the thesis.
        
        Args:
            i (int): Index of the data point.

        Returns:
            np.ndarray: Log posterior predictive probabilities for each cluster.
        """
        k_Ns = self.prior.k_0 + self.counts[:self.K]
        v_Ns = self.prior.v_0 + self.counts[:self.K]
        m_Ns = (self.sum_X[:self.K] + self._cache_prod_k0m0) / k_Ns[:, np.newaxis]
        Vs = v_Ns - self.D + 1

        deltas = m_Ns - self.X[i]
        mahalonabis_dist = np.zeros(self.K)

        for k in range(self.K):
            mahalonabis_dist[k] = np.matmul(np.matmul(deltas[k], self.inv_covariances[k]), deltas[k])

        prob = np.zeros(self.K_max)
        prob[:self.K] = (
            self._cache_gammaln_by_2[Vs + self.D] - self._cache_gammaln_by_2[Vs]
            - (self.D / 2) * (self._cache_log_Vs[Vs] + self._cache_log_pi)
            - 0.5 * self.log_det_covariances[:self.K]
            - 0.5 * (Vs + self.D) * np.log(1 + mahalonabis_dist / Vs)
        )

        prob[self.K:] = self.log_post_pred_prior(i)
        return prob

    def get_post_S_N(self, k: int):
        """
        Returns the posterior hyperparameters for the k-th cluster.
        Equation 19 in the thesis.

        Args:
            k (int): Index of the cluster.

        Returns:
            np.ndarray: Posterior hyperparameters S_N.
        """
        k_N = self.prior.k_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        S_N = self._cache_partial_S_sum + self.outer_prod_X[k] - k_N * np.outer(m_N, m_N)
        return S_N

    def get_post_posterior_S_N(self, k: int):
        """
        Returns the posterior posterior hyperparameters for the k-th cluster.

        Args:
            k (int): Index of the cluster.

        Returns:
            np.ndarray: Posterior posterior hyperparameters S_N.
        """
        k_N = self.prior.k_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N

        post_k_N = k_N + self.counts[k]
        post_m_N = (self.sum_X[k] + m_N * k_N) / post_k_N
        post_S_N = (
            self.prior.S_0 + self._cache_prior_outer_m_0 + 2 * self.outer_prod_X[k] 
            - post_k_N * np.outer(post_m_N, post_m_N)
        )
        return post_S_N

    def get_posterior_probability_Z_k(self, k: int):
        """
        Returns the posterior probability for the k-th cluster.
        Equation 20 in the thesis.

        Args:
            k (int): Index of the cluster.

        Returns:
            float: Posterior probability.
        """
        if k >= self.K:
            return gammaln(self.alpha / self.K_max)
        else:
            v_N = self.prior.v_0 + self.counts[k]
            post_v_N = self.prior.v_0 + 2 * self.counts[k]
            S_N = self.get_post_S_N(k)
            post_S_N = self.get_post_posterior_S_N(k)
            log_post_Z = (
                -1 * self.counts[k] * (self.D / 2) * self._cache_log_pi
                + self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N]
                + (v_N / 2) * np.log(slogdet(S_N)[1]) - (post_v_N / 2) * np.log(slogdet(post_S_N)[1])
                - (self.D / 2) * (self._cache_log_Ks[self.counts[k]] - self._cache_log_Ks[2 * self.counts[k]])
                + gammaln(self.alpha / self.K_max + self.counts[k])
            )
            return log_post_Z

    def get_log_marginal(self, k: int):
        """
        Returns the log marginal likelihood for the k-th cluster.
        Equation 20 in the thesis.

        Args:
            k (int): Index of the cluster.

        Returns:
            float: Log marginal likelihood.
        """
        k_N, v_N, m_N, S_N = self.get_post_hyperparameters(k)
        gammas = [
            gammaln((v_N + 1 - i) / 2) - gammaln((self.prior.v_0 + 1 - i) / 2)
            for i in range(1, self.D + 1)
        ]
        return (
            (-1 * self.counts[k] * self.D / 2) * self._cache_log_pi
            + (self.D / 2) * (np.log(self.prior.k_0) - np.log(self.k_N))
            + self.prior.v_0 * np.log(slogdet(self.prior.S_0[k])[1])
            - v_N * np.log(slogdet(S_N[k])[1])
            + gammas
        )

    def random_cluster_params(self, k: int):
        """
        Returns random mean vector and covariance matrix from the posterior NIW distribution for cluster k.
        Equation 19 in the thesis.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Random mean vector (np.ndarray) and covariance matrix (np.ndarray).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        S_N = self.S_N_partials[k] - k_N * np.outer(m_N, m_N)

        sigma = invwishart.rvs(df=v_N, scale=S_N)

        if self.D == 1:
            mu = np.random.normal(m_N, sigma / k_N)
        else:
            mu = np.random.multivariate_normal(m_N, sigma / k_N)
        return mu, sigma

    def map_cluster_params(self, k: int):
        """
        Returns the MAP estimates of the mean vector and covariance matrix for cluster k.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Mean vector (np.ndarray) and covariance matrix (np.ndarray).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        sigma = (self._cache_partial_S_sum + self.outer_prod_X[k] - k_N * np.outer(m_N, m_N)) / (v_N + self.D + 2)
        return m_N, sigma

    def _multivariate_students_t_prior(self, i: int):
        """
        Computes the log of the multivariate Student's t-distribution prior probability for the i-th data point.

        Args:
            i (int): Index of the data point.

        Returns:
            float: Log probability under the multivariate Student's t-distribution prior.
        """
        mu = self.prior.m_0
        inv_covariance = self._cache_inv_covariance_prior
        v = self.prior.v_0 - self.D + 1

        delta = self.X[i, :] - mu
        return (
            self._cache_post_pred_prior_coeff
            - (v + self.D) / 2.0 * np.log(1 + 1.0 / v * np.dot(np.dot(delta, inv_covariance), delta))
        )

    def _update_log_det_covariance_and_inv_covariance_priors(self, k: int):
        """
        Updates the log determinant and inverse covariance matrix for the prior of the k-th cluster.

        Args:
            k (int): Index of the cluster.
        """
        self.log_det_covariances[k] = self._cache_logdet_covariance_prior
        self.inv_covariances[k, :, :] = self._cache_inv_covariance_prior

    def _update_log_det_covariance_and_inv_covariance(self, k: int):
        """
        Updates the log determinant and inverse covariance matrix for the k-th cluster.
        Refer to Equation 20 in the thesis.
        
        Args:
            k (int): Index of the cluster.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N

        covar = (
            (k_N + 1.0) / (k_N * (v_N - self.D + 1.0))
            * (self._cache_partial_S_sum + self.outer_prod_X[k] - k_N * np.outer(m_N, m_N))
        )
        self.log_det_covariances[k] = slogdet(covar)[1]
        self.inv_covariances[k, :, :] = inv(covar)


# class for statistics of the Gaussian clusters with diagonal covariance matrix
class gaussianClustersDiag:
    """
    Class for managing Gaussian clusters with diagonal covariance matrices.

    Attributes:
        X (np.ndarray): Data matrix of shape (N, D).
        prior (object): Prior distribution parameters.
        alpha (float): Dirichlet process concentration parameter.
        K_max (int): Maximum number of clusters.
        assignments (np.ndarray): Initial assignments of clusters.
        sum_X (np.ndarray): Sum of data points for each cluster.
        square_prod_X (np.ndarray): Sum of squares of data points for each cluster.
        counts (np.ndarray): Number of data points in each cluster.
        log_det_covariances (np.ndarray): Log determinants of covariance matrices.
        inv_covariances (np.ndarray): Inverse covariance matrices for each cluster.
    """

    def __init__(self, X, prior, alpha, K_max, assignments=None):
        """
        Initialize a gaussianClustersDiag object.

        Args:
            X (np.ndarray): Data matrix of shape (N, D).
            prior (object): Prior distribution parameters.
            alpha (float): Dirichlet process concentration parameter.
            K_max (int): Maximum number of clusters.
            assignments (np.ndarray, optional): Initial assignments of clusters. Defaults to None.
        """
        self.X = X
        self.N, self.D = X.shape
        self.prior = prior
        self.alpha = alpha
        self.K_max = K_max
        
        # Initialize arrays for cluster statistics
        self.sum_X = np.zeros((self.K_max, self.D), dtype=float)
        self.square_prod_X = np.zeros((self.K_max, self.D), dtype=float)
        self.counts = np.zeros(self.K_max, dtype=int)
        
        # Initialize arrays for hyperparameters
        self.log_det_covariances = np.zeros(self.K_max, dtype=float)
        self.inv_covariances = np.zeros((self.K_max, self.D), dtype=float)
        
        # Cache some values for efficiency
        self._cache()
        
        # Initialize number of clusters
        self.K = 0
        
        # Assign initial cluster assignments
        self.assignments = assignments
        if assignments is not None:
            for k in range(self.assignments.max() + 1):
                for i in np.where(self.assignments == k)[0]:
                    self.add_assignment(i, k)
    
    def _cache(self):
        """
        Pre-compute and cache values to avoid redundant computations.
        """
        self._cache_square_X = np.square(self.X)
        self._cache_prior_square_m_0 = np.square(self.prior.m_0)
        
        Ns = np.concatenate([[1], np.arange(1, self.prior.v_0 + 2 * self.N + 4)])
        self._cache_gammaln_by_2 = gammaln(Ns / 2.0)
        self._cache_log_pi = np.log(np.pi)
        self._cache_log_Vs = np.log(Ns)
        self._cache_gammaln_alpha = gammaln(self.alpha)
        self._cache_prod_k0m0 = self.prior.k_0 * self.prior.m_0
        self._cache_partial_S_sum = self.prior.S_0 + self.prior.k_0 * np.square(self.prior.m_0)

        var = self.prior.S_0 * (self.prior.k_0 + 1.0) / (self.prior.k_0 * self.prior.v_0)
        self._cache_inv_var_prior = 1.0 / var
        self._cache_log_var_prod_prior = 0.5 * np.log(var).sum()
        self._cache_post_pred_coeff_prior = (
            self.D
            * (
                self._cache_gammaln_by_2[self.prior.v_0 + 1]
                - self._cache_gammaln_by_2[self.prior.v_0]
                - 0.5 * self._cache_log_Vs[self.prior.v_0]
                - 0.5 * self._cache_log_pi
            )
            - self._cache_log_var_prod_prior
        )
    
    def cache_cluster_stats(self, k):
        """
        Cache statistics of cluster k.

        Args:
            k (int): Cluster index.

        Returns:
            tuple: Cached statistics of the cluster (log_det_covariance, inv_covariance, count, sum_X, square_prod_X).
        """
        return (
            self.log_det_covariances[k].copy(),
            self.inv_covariances[k].copy(),
            self.counts[k].copy(),
            self.sum_X[k].copy(),
            self.square_prod_X[k].copy()
        )
    
    def restore_cluster_stats(self, k, log_det_covariance, inv_covariance, count, sum_X, square_prod_X):
        """
        Restore cached statistics for cluster k.

        Args:
            k (int): Cluster index.
            log_det_covariance (float): Log determinant of covariance.
            inv_covariance (np.ndarray): Inverse covariance matrix.
            count (int): Number of data points in the cluster.
            sum_X (np.ndarray): Sum of data points in the cluster.
            square_prod_X (np.ndarray): Sum of squares of data points in the cluster.
        """
        self.log_det_covariances[k] = log_det_covariance
        self.inv_covariances[k] = inv_covariance
        self.counts[k] = count
        self.sum_X[k] = sum_X
        self.square_prod_X[k] = square_prod_X
    
    def add_assignment(self, i, k):
        """
        Add assignment of data point i to cluster k.

        Args:
            i (int): Data point index.
            k (int): Cluster index.
        """
        if k == self.K:
            self.K += 1
            self.sum_X[k, :] = np.zeros(self.D)
            self.square_prod_X[k, :] = np.zeros(self.D)
        
        self.assignments[i] = k
        self.sum_X[k, :] += self.X[i]
        self.square_prod_X[k, :] += self._cache_square_X[i]
        self.counts[k] += 1
        self._update_log_det_covariance_and_inv_covariance(k)
    
    def del_assignment(self, i):
        """
        Delete assignment of data point i from its cluster.

        Args:
            i (int): Data point index.
        """
        k = self.assignments[i]
        
        if k != -1:
            self.assignments[i] = -1
            self.counts[k] -= 1
            if self.counts[k] == 0:
                self.empty_cluster(k)
            else:
                self.sum_X[k, :] -= self.X[i]
                self.square_prod_X[k, :] -= self._cache_square_X[i]
                self._update_log_det_covariance_and_inv_covariance(k)
    
    def empty_cluster(self, k):
        """
        Remove an empty cluster and adjust the cluster statistics.

        Args:
            k (int): Cluster index to be removed.
        """
        self.K -= 1
        if k != self.K:
            self.sum_X[k, :] = self.sum_X[self.K, :]
            self.square_prod_X[k, :] = self.square_prod_X[self.K, :]
            self.counts[k] = self.counts[self.K]
            self.log_det_covariances[k] = self.log_det_covariances[self.K]
            self.inv_covariances[k, :] = self.inv_covariances[self.K, :]
            self.assignments[np.where(self.assignments == self.K)] = k
        
        self.counts[self.K] = 0
        self._update_log_det_covariance_and_inv_covariance_priors(self.K)
        self.sum_X[self.K, :] = np.zeros(self.D)
        self.square_prod_X[self.K, :] = np.zeros(self.D)
    
    def log_post_pred(self, i):
        """
        Compute log posterior predictive probability for data point i.
        Refer to equation 32 in the thesis.

        Args:
            i (int): Data point index.

        Returns:
            np.ndarray: Log posterior predictive probabilities for all clusters.
        """
        k_Ns = self.prior.k_0 + self.counts[:self.K]
        v_Ns = self.prior.v_0 + self.counts[:self.K]
        m_Ns = (self.sum_X[:self.K] + self._cache_prod_k0m0) / k_Ns[:, np.newaxis]
        deltas = m_Ns - self.X[i]
        
        res = np.zeros(self.K_max)
        res[:self.K] = self.D * (
            self._cache_gammaln_by_2[v_Ns + 1] - self._cache_gammaln_by_2[v_Ns]
            - 0.5 * self._cache_log_Vs[v_Ns] - 0.5 * self._cache_log_pi
        ) - 0.5 * self.log_det_covariances[:self.K] - (v_Ns + 1) / 2.0 * (
            np.log(1 + np.square(deltas) * self.inv_covariances[:self.K] * (1.0 / v_Ns[:, np.newaxis]))
        ).sum(axis=1)

        res[self.K:] = self._students_t_prior(i)
        return res
    
    def _update_log_det_covariance_and_inv_covariance_priors(self, k):
        """
        Update log determinant of covariance and inverse covariance matrix for prior.
        
        Args:
            k (int): Cluster index.
        """
        self.log_det_covariances[k] = self._cache_log_var_prod_prior
        self.inv_covariances[k, :] = self._cache_inv_var_prior
    
    def _update_log_det_covariance_and_inv_covariance(self, k):
        """
        Update log determinant of covariance and inverse covariance matrix for cluster k.
        Refer to equation 30 in the thesis.
        
        Args:
            k (int): Cluster index.
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N

        var = (k_N + 1) / (k_N * v_N) * (self._cache_partial_S_sum + self.square_prod_X[k] - k_N * np.square(m_N))
        self.log_det_covariances[k] = np.log(var).sum()
        self.inv_covariances[k, :] = 1.0 / var
    
    def get_post_hyperparameters(self, k):
        """
        Get posterior hyperparameters for cluster k.
        Refer to equation 30 in the thesis.
        
        Args:
            k (int): Cluster index.
        
        Returns:
            tuple: Posterior hyperparameters (k_N, v_N, m_N, S_N).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        S_N = self._cache_partial_S_sum + self.square_prod_X[k] - k_N * np.square(m_N)
        return k_N, v_N, m_N, S_N
    
    def get_post_posterior_hyperparameters(self, k):
        """
        Get posterior predictive hyperparameters for cluster k.
        
        Args:
            k (int): Cluster index.
        
        Returns:
            tuple: Posterior predictive hyperparameters (post_k_N, post_v_N, post_m_N, post_S_N).
        """
        
        k_N = self.prior.k_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        post_k_N = k_N + self.counts[k]
        post_v_N = self.prior.v_0 + 2 * self.counts[k]
        post_m_N = (self.sum_X[k] + m_N * k_N) / post_k_N
        post_S_N = self.prior.S_0 + self._cache_prior_square_m_0 + 2 * self.square_prod_X[k] - post_k_N * np.square(post_m_N)
        return post_k_N, post_v_N, post_m_N, post_S_N
    
    def get_posterior_probability_Z_k(self, k):
        """
        Compute logarithm of posterior probability of cluster k.
        Refer to equation 31 in the thesis.
        
        Args:
            k (int): Cluster index.
        
        Returns:
            float: Logarithm of posterior probability of cluster k.
        """
        if k >= self.K:
            return gammaln(self.alpha / self.K_max)
        else:
            k_N, v_N, m_N, S_N = self.prior.k_0, self.prior.v_0, self.prior.m_0, self.prior.S_0
            post_k_N, post_v_N, post_m_N, post_S_N = self.get_post_hyperparameters(k)
            
            log_post_Z = (
                self.D * (
                    (-1.0 * self.counts[k] / 2) * self._cache_log_pi
                    + self._cache_gammaln_by_2[post_v_N]
                    - self._cache_gammaln_by_2[v_N]
                    + (1.0 / 2) * (np.log(k_N) - np.log(post_k_N))
                )
                + np.sum((v_N / 2) * np.log(S_N) - (post_v_N / 2) * np.log(post_S_N))
                + gammaln(self.alpha / self.K_max + self.counts[k])
            )
            return log_post_Z
    
    def random_cluster_params(self, k):
        """
        Generate random mean vector and covariance matrix from the posterior NIW distribution for cluster k.
        Refer to equation 31 in the thesis.
        
        Args:
            k (int): Cluster index.
        
        Returns:
            tuple: Random mean vector and covariance matrix (mu, sigma).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        S_N = self._cache_partial_S_sum[k] - k_N * np.outer(m_N, m_N)
        
        sigma = invwishart.rvs(df=v_N, scale=S_N)
        
        if self.D == 1:
            mu = np.random.normal(m_N, sigma / k_N)
        else:
            mu = np.random.multivariate_normal(m_N, sigma / k_N)
        
        return mu, sigma
    
    def map_cluster_params(self, k):
        """
        Compute MAP estimates of cluster's mean vector and covariance matrix.
        
        Args:
            k (int): Cluster index.
        
        Returns:
            tuple: MAP estimates of mean vector and covariance matrix (m_N, sigma).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        sigma = (self._cache_partial_S_sum + self.outer_prod_X[k] - k_N * np.outer(m_N, m_N)) / (v_N + self.D + 2)
        return m_N, sigma
    
    def _students_t_prior(self, i):
        """
        Compute log prior probability of data point i under Student's t-distribution prior.
        
        Args:
            i (int): Data point index.
        
        Returns:
            float: Log prior probability of data point i.
        """
        inv_var = self._cache_inv_var_prior
        v = self.prior.v_0
        mu = self.prior.m_0
        delta = self.X[i, :] - mu
        
        return (
            self._cache_post_pred_coeff_prior
            - ((v + 1.0) / 2.0 * (np.log(1.0 + 1.0 / v * np.square(delta) * inv_var)).sum())
        )


# class for statistics of the categorical clusters
class categoricalClusters(object):
    """
    Class for handling categorical data clustering using a specified prior.
    
    Attributes:
        N (int): Number of data points.
        D (int): Number of dimensions (attributes) in the categorical data.
        alpha (float): Dirichlet process parameter.
        C (numpy.ndarray): Categorical data matrix of shape (N, D).
        gamma (float): Hyperparameter for the Dirichlet distribution.
        K_max (int): Maximum number of clusters including the empty ones.
        Ms (numpy.ndarray): Array containing unique counts of categories for each dimension.
        counts (numpy.ndarray): Array to store counts of data points assigned to each cluster.
        catCounts (numpy.ndarray): Array to store categorical counts for each cluster and dimension.
        assignments (numpy.ndarray): Initial assignments of data points to clusters.
        K (int): Current number of clusters.
    """

    def __init__(self, C, alpha, gamma, K, assignments=None):
        """
        Initialize categorical clustering instance.
        
        Args:
            C (numpy.ndarray): Categorical data matrix of shape (N, D).
            alpha (float): Dirichlet process parameter.
            gamma (float): Hyperparameter for the Dirichlet distribution.
            K (int): Maximum number of clusters including the empty ones.
            assignments (numpy.ndarray or None): Initial assignments of data points to clusters.
                If None, no initial assignments are made.
        """
        self.N, self.D = C.shape
        self.alpha = alpha
        self.C = C
        self.gamma = gamma
        self.K_max = K

        self.Ms = np.zeros(self.D, int)
        for d in range(self.D):
            self.Ms[d] = len(set(C[:, d]))

        self.counts = np.zeros(self.K_max, int)
        self.catCounts = np.zeros((self.K_max, self.Ms.max(), self.D), int)

        self._cache()

        self.K = 0

        if assignments is None:
            self.assignments = -1 * np.ones(self.N, int)
        else:
            self.assignments = assignments
            for k in range(self.assignments.max() + 1):
                for i in np.where(self.assignments == k)[0]:
                    self.add_assignment(i, k)

    def _cache(self):
        """
        Cache some precomputed values for efficiency.
        """
        self._cache_log_pi = np.log(np.pi)
        self._cache_gammaln_alpha = gammaln(self.alpha)

    def cache_cluster_stats(self, k):
        """
        Cache cluster k's statistics.

        Args:
            k (int): Cluster index.

        Returns:
            tuple: Cached cluster statistics (counts, catCounts).
        """
        return (
            self.counts[k].copy(),
            self.catCounts[k].copy()
        )

    def restore_cluster_stats(self, k, count_N, catCount_N):
        """
        Restore cluster k's statistics.

        Args:
            k (int): Cluster index.
            count_N (numpy.ndarray): New counts for cluster k.
            catCount_N (numpy.ndarray): New categorical counts for cluster k.
        """
        self.counts[k] = count_N
        self.catCounts[k] = catCount_N

    def add_assignment(self, i, k):
        """
        Assign data point i to cluster k.

        Args:
            i (int): Data point index.
            k (int): Cluster index.
        """
        if k == self.K:
            self.K += 1

        self.assignments[i] = k

        for d in range(self.D):
            self.catCounts[k][self.C[i][d]][d] += 1

        self.counts[k] += 1

    def del_assignment(self, i):
        """
        Delete assignment of data point i.

        Args:
            i (int): Data point index.
        """
        k = self.assignments[i]

        if k != -1:
            self.assignments[i] = -1
            self.counts[k] -= 1
            for d in range(self.D):
                self.catCounts[k][self.C[i][d]][d] -= 1

            if self.counts[k] == 0:
                self.empty_cluster(k)

    def empty_cluster(self, k):
        """
        Empty out cluster k.

        Args:
            k (int): Cluster index.
        """
        self.K -= 1
        if k != self.K:
            self.counts[k] = self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k
            self.catCounts[k, :] = self.catCounts[self.K, :]

        self.counts[self.K] = 0
        self.catCounts[self.K, :] = np.zeros((self.Ms.max(), self.D), int)

    def log_post_pred(self, i):
        """
        Compute log posterior predictive probability for data point i.
        Refer to equation 38 in the thesis.

        Args:
            i (int): Data point index.

        Returns:
            numpy.ndarray: Array of log posterior predictive probabilities for each cluster.
        """
        gamma_Ns = self.gamma + self.catCounts

        res = np.zeros((self.K_max, self.D))
        for d in range(self.D):
            res[:, d] = np.log(gamma_Ns[:, self.C[i][d], d]) - np.log(self.counts + self.gamma * self.Ms[d])

        return res.sum(axis=1)

    def get_posterior_probability_Z_k(self, k):
        """
        Compute log posterior probability of cluster k.
        Refer to equation 37 in the thesis.

        Args:
            k (int): Cluster index.

        Returns:
            float: Log posterior probability of cluster k.
        """
        gamma_N = self.gamma + self.catCounts[k]
        post_gamma_N = gamma_N + self.catCounts[k]

        log_post_Z = np.zeros((self.D))
        for d in range(self.D):
            log_post_Z[d] = gammaln(gamma_N[:, d].sum()) + gammaln(post_gamma_N[:, d]).sum() \
                             - gammaln(gamma_N[:, d]).sum() - gammaln(self.counts[k] + gamma_N[:, d].sum())

        return log_post_Z.sum() + gammaln(self.alpha / self.K_max + self.counts[k])

    
    # return 0
    
        if k >= self.K:
            return gammaln(self.alpha/self.K_max) 
        
        else:
            gamma_N = self.gamma + self.catCounts[k]
            log_post_Z = np.zeros((self.D))
            for d in range(self.D):
                # log_post_Z[d] = gammaln(self.gamma * self.Ms[d]) +  gammaln(gamma_N[:, d]).sum() - self.Ms[d] * gammaln(self.gamma) -  gammaln(self.gamma * self.Ms[d] + self.counts[k])
                log_post_Z[d] = gammaln(gamma_N[:, d]).sum() -  gammaln(self.gamma * self.Ms[d] + self.counts[k])


            return log_post_Z.sum()
    
    # def random_cluster_params(self, k):    
    #     return 1
    
    # def map_cluster_params(self, k):
    #    return 1

    # class for statistics of the clusters


        
# class for the statistics of the mixed data with categorical and Gaussian features combined
class categoricalGaussianClusters(object):
    """
    Class for managing clusters of mixed categorical and Gaussian data.

    This class handles the statistics and operations for clusters containing
    both categorical and Gaussian features. It manages cluster assignments,
    updates cluster statistics, and computes various probabilities.

    Attributes:
        X (numpy.ndarray): Input data matrix for Gaussian features.
        C (numpy.ndarray): Input data matrix for categorical features.
        N (int): Number of data points.
        gD (int): Number of Gaussian features.
        cD (int): Number of categorical features.
        gamma (float): Concentration prior hyperparameter for categorical distribution.
        prior (object): Prior hyperparameter for Gaussian features.
        K_max (int): Maximum number of clusters.
        alpha (float): Concentration prior hyperparameter for the Dirichlet process.
        Ms (numpy.ndarray): Number of categories for each categorical feature.
        sum_X (numpy.ndarray): Sum of Gaussian data for each cluster.
        square_prod_X (numpy.ndarray): Sum of squared Gaussian data for each cluster.
        counts (numpy.ndarray): Number of data points in each cluster.
        catCounts (numpy.ndarray): Array to store categorical counts for each cluster and dimension.
        log_det_covariances (np.ndarray): Log determinants of covariance matrices.
        inv_covariances (np.ndarray): Inverse covariance matrices for each cluster.
        K (int): Current number of non-empty clusters.
        assignments (numpy.ndarray): Cluster assignments for each data point.
    """

    def __init__(self, X, C, alpha, prior, gamma, K, assignments=None):
        """
        Initialize the categoricalGaussianClusters object.

        Args:
            X (numpy.ndarray): Input data matrix for Gaussian features.
            C (numpy.ndarray): Input data matrix for categorical features.
            alpha (float): Concentration parameter for the Dirichlet process.
            prior (object): Prior distribution for Gaussian features.
            gamma (float): Hyperparameter for categorical distribution.
            K (int): Maximum number of clusters.
            assignments (numpy.ndarray, optional): Initial cluster assignments.
        """
        # assignments is initial assignments of clusters
        # K-max is the maximum number of clusters including the empty ones

        self.X = X
        self.C = C
        self.N, self.gD = X.shape
        self.N, self.cD = C.shape
        self.gamma = gamma
        self.prior = prior
        self.K_max = K
        self.alpha = alpha
        
        self.Ms = np.zeros(self.cD, int)
        for d in range(self.cD):
            self.Ms[d] = len(set(C[:, d]))
        
        ####### partial_hyperparaneters_attr #############
        self.sum_X = np.zeros((self.K_max, self.gD), float)
        self.square_prod_X = np.zeros((self.K_max, self.gD), float)
        self.counts = np.zeros(self.K_max, int)

        self.catCounts = np.zeros((self.K_max, self.Ms.max(), self.cD), int) 
        ####### hyper-parameters' attributes initialization ########

        # log of determinant of multivariate Student's t distribution associated with each of the K cluster (kx1 vector)
        self.log_det_covariances = np.zeros(self.K_max)
        
        # inverse of S_N_partials (Kx(DxD) matrix)
        self.inv_covariances = np.zeros((self.K_max, self.gD))

        # to avoid recomputing we will cache some log and log gamma values
        self._cache()

        # Initialization
        self.K = 0

        # assign the initial assignments
        self.assignments = assignments
        
        # adding the assigned clusters
        for k in range(self.assignments.max() + 1):
            for i in np.where(self.assignments == k)[0]:
                self.add_assignment(i, k)


    def _cache(self):
        """
        Precompute and cache frequently used values to improve performance.
        """
        # pre-computing outer products
        # self._cache_square_X = np.zeros((self.N, self.D))
        self._cache_square_X = np.square(self.X) 
        
        self._cache_prior_square_m_0 = np.square(self.prior.m_0)

        # pre-computing gamma values of possible numbers (for computing student's t)
        Ns = np.concatenate([[1], np.arange(1, self.prior.v_0 + 2* self.N + 4)])
        self._cache_gammaln_by_2 = gammaln(Ns/2.)
        self._cache_log_pi = np.log(np.pi)
        self._cache_log_Vs = np.log(Ns)
        self._cache_gammaln_alpha = gammaln(self.alpha)
        self._cache_prod_k0m0 = self.prior.k_0 * self.prior.m_0
        self._cache_partial_S_sum = self.prior.S_0 + self.prior.k_0*np.square(self.prior.m_0)

        var = (self.prior.k_0 + 1.) / (self.prior.k_0*self.prior.v_0) * self.prior.S_0
        self._cache_inv_var_prior = 1./var
        self._cache_log_var_prod_prior = 0.5*np.log(var).sum()
        self._cache_post_pred_coeff_prior = self.gD * ( self._cache_gammaln_by_2[self.prior.v_0 + 1] - self._cache_gammaln_by_2[self.prior.v_0]
                - 0.5*self._cache_log_Vs[self.prior.v_0] - 0.5*self._cache_log_pi
                ) - self._cache_log_var_prod_prior

    def cache_cluster_stats(self, k):
        """
        Cache the statistics of a specific cluster.

        Args:
            k (int): Cluster index.

        Returns:
            tuple: Cached statistics for the cluster.
        """
        # caching cluster k's statistics in a tuple
        return (
            self.log_det_covariances[k].copy(),
            self.inv_covariances[k].copy(),
            self.counts[k].copy(),
            self.sum_X[k].copy(),
            self.square_prod_X[k].copy(),
            self.catCounts[k].copy()
            )


    def restore_cluster_stats(self, k, log_det_covariance, inv_covariance, count, sum_X, outer_prod_X, catCount_N):
        """
        Restore the statistics of a specific cluster.

        Args:
            k (int): Cluster index.
            log_det_covariance (float): Log determinant of covariance.
            inv_covariance (numpy.ndarray): Inverse of covariance.
            count (int): Number of data points in the cluster.
            sum_X (numpy.ndarray): Sum of Gaussian features.
            outer_prod_X (numpy.ndarray): Sum of squared Gaussian features.
            catCount_N (numpy.ndarray): Counts of categorical features.
        """
        # restore the cluster stats for the attributes
        self.log_det_covariances[k] = log_det_covariance
        self.inv_covariances[k, :] = inv_covariance
        self.counts[k] = count
        self.sum_X[k] = sum_X
        self.square_prod_X[k] = outer_prod_X

        self.catCounts[k] = catCount_N

    def add_assignment(self, i, k):
        """
        Add a data point to a cluster.

        Args:
            i (int): Index of the data point.
            k (int): Index of the cluster.
        """
        # assigning new cluster k for the ith observation
        if k == self.K:
            self.K += 1
            
            # initializing the partial attributes for new k
            self.sum_X[k, :] = np.zeros(self.gD)
            self.square_prod_X[k, :] = np.zeros(self.gD)


        self.assignments[i] = k

        # updating the partial hyperparameters
        for d in range(self.cD):
            self.catCounts[k][self.C[i][d]][d] += 1

        self.sum_X[k, :] += self.X[i]
        self.square_prod_X[k, :] += self._cache_square_X[i]
        self.counts[k] += 1

        # updating covariance matrix attributes
        self._update_log_det_covariance_and_inv_covariance(k) 


    def del_assignment(self, i):
        """
        Delete assignment of data point i from its cluster.

        Args:
            i (int): Data point index.
        """
        # delete the assignment and attributes of i-th data vector
        k = self.assignments[i]

        if k != -1 :
            self.assignments[i] = -1
            self.counts[k] -= 1

            for d in range(self.cD):
                self.catCounts[k][self.C[i][d]][d] -= 1

            if self.counts[k] == 0:
                
                # if cluster is empty, remove it
                self.empty_cluster(k)
            else:

                # update attributions
                self.sum_X[k, :] -= self.X[i]
                self.square_prod_X[k, :] -= self._cache_square_X[i]
                
                self._update_log_det_covariance_and_inv_covariance(k)


    def empty_cluster(self, k):
        """
        Remove an empty cluster and adjust the cluster statistics.

        Args:
            k (int): Index of the empty cluster to be removed.
        """
        self.K -= 1
        if k != self.K:

            # put all stats from last cluster into the empty cluster (one which is being remopved)
            self.sum_X[k, :] = self.sum_X[self.K, :]
            self.square_prod_X[k, :] = self.square_prod_X[self.K, :]
            self.counts[k] = self.counts[self.K]
            self.log_det_covariances[k] = self.log_det_covariances[self.K]
            self.inv_covariances[k, :] = self.inv_covariances[self.K, :]
            self.assignments[np.where(self.assignments == self.K)] = k

            self.catCounts[k, :] = self.catCounts[self.K, :]

        # # empty out stats from last cluster
        # self.log_det_covariances[self.K] = 0.
        # self.inv_covariances[self.K, :, :].fill(0.)
        # self.counts[self.K] = 0

        # fill out priors stats from last cluster
        self.counts[self.K] = 0
        self._update_log_det_covariance_and_inv_covariance_priors(self.K)

        self.catCounts[self.K, :] = np.zeros((self.Ms.max(), self.cD), int)

        self.sum_X[self.K, :] = np.zeros(self.gD)
        self.square_prod_X[self.K, :] = np.zeros(self.gD)

    
    def log_post_pred_gauss(self, i):
        """
        Compute the log posterior predictive probability for Gaussian features.
        Refer to equation 32 in the thesis.

        Args:
            i (int): Index of the data point.

        Returns:
            numpy.ndarray: Log posterior predictive probabilities for each cluster.
        """
        
        # for j in range(self.K_max):
        #     if self.counts[0] != self.assignments.tolist().count(0):
        #         print("f*cked up")

        # returns k dimension vector student's t pdf
        k_Ns = self.prior.k_0 + self.counts[:self.K]
        v_Ns = self.prior.v_0 + self.counts[:self.K]
        m_Ns = (self.sum_X[:self.K] + self._cache_prod_k0m0)/k_Ns[:, np.newaxis]

        
        deltas = m_Ns - self.X[i]


        res = np.zeros(self.K_max)
        res[:self.K] =  self.gD * (
                self._cache_gammaln_by_2[v_Ns + 1] - self._cache_gammaln_by_2[v_Ns] 
                - 0.5*self._cache_log_Vs[v_Ns] - 0.5*self._cache_log_pi
                ) - 0.5*self.log_det_covariances[:self.K] - (v_Ns + 1)/2. * (np.log(
                1 + np.square(deltas)*self.inv_covariances[:self.K]*(1./v_Ns[:, np.newaxis])
                )).sum(axis=1)

        # S_Ns = (1+1/k_Ns)[:, np.newaxis]*(self._cache_partial_S_sum + self.square_prod_X[:self.K] - k_Ns[:, np.newaxis]*np.square(m_Ns))
        # res[:self.K] =  self.gD * (
        #         self._cache_gammaln_by_2[v_Ns + 1] - self._cache_gammaln_by_2[v_Ns] 
        #         - 0.5*self._cache_log_Vs[v_Ns] - 0.5*self._cache_log_pi
        #         ) - 0.5*np.sum(np.log(S_Ns), axis=1) - (v_Ns + 1)/2. * (np.log(
        #         1 + np.square(deltas)*(1/S_Ns)*(1./v_Ns[:, np.newaxis])
        #         )).sum(axis=1)

        # breakpoint()

        res[self.K:] = self._students_t_prior(i)

        return res

    def log_post_pred_cat(self, i):
        """
        Compute the log posterior predictive probability for categorical features.
        Refer to equation 38 in the thesis.
        
        Args:
            i (int): Index of the data point.

        Returns:
            numpy.ndarray: Log posterior predictive probabilities for each cluster.
        """
        gamma_Ns = self.gamma + self.catCounts

        res = np.zeros((self.K_max, self.cD))
        for d in range(self.cD):
            res[:,d] = np.log(gamma_Ns[:, self.C[i][d], d]) - np.log(self.counts + self.gamma*self.Ms[d])
        
        return res.sum(axis=1)

    def _update_log_det_covariance_and_inv_covariance_priors(self, k):
        """
        Update the log determinant of covariance and inverse covariance for priors.

        Args:
            k (int): Index of the cluster.
        """
        self.log_det_covariances[k] = self._cache_log_var_prod_prior
        self.inv_covariances[k, :] = self._cache_inv_var_prior

    def _update_log_det_covariance_and_inv_covariance(self, k):
        """
        Update the log determinant of covariance and inverse covariance for a cluster.

        Args:
            k (int): Index of the cluster to update.
        """
        # Update the log_det_covariance and inv_covariance for cluster k       
        k_N = self.prior.k_0 + self.counts[k] 
        v_N = self.prior.v_0 + self.counts[k] 
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N 

        # Construct covariance matrix, (S_N = S_N_partials - k_N*m_M*m_N')
        var = (k_N + 1) / (k_N * v_N) * (self._cache_partial_S_sum + self.square_prod_X[k] - k_N * np.square(m_N))

        self.log_det_covariances[k] = np.log(var).sum()

        # Inverse of covariance matrix
        self.inv_covariances[k, :] = 1. / var

    def get_post_hyperparameters_gauss(self, k):
        """
        Get posterior hyperparameters for Gaussian distribution of a cluster.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Posterior hyperparameters (k_N, v_N, m_N, S_N).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        S_N = self._cache_partial_S_sum + self.square_prod_X[k] - k_N * np.square(m_N)
        return k_N, v_N, m_N, S_N
    
    def get_post_posterior_hyperparameters_gauss(self, k):
        """
        Get posterior hyperparameters for Gaussian distribution of a cluster.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Posterior hyperparameters (post_k_N, post_v_N, post_m_N, post_S_N).
        """
        k_N = self.prior.k_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N

        post_k_N = k_N + self.counts[k]
        post_v_N = self.prior.v_0 + 2 * self.counts[k]

        post_m_N = (self.sum_X[k] + m_N * k_N) / post_k_N
        post_S_N = self.prior.S_0 + self._cache_prior_square_m_0 + 2 * self.square_prod_X[k] - post_k_N * np.square(post_m_N)
        return post_k_N, post_v_N, post_m_N, post_S_N

    def get_posterior_probability_Z_k_gauss(self, k):
        """
        Compute the posterior probability of Z_k for Gaussian features.

        Args:
            k (int): Index of the cluster.

        Returns:
            float: Log posterior probability of Z_k for Gaussian features.
        """
        k_N, v_N, m_N, S_N = self.get_post_hyperparameters_gauss(k)
        
        log_post_Z = self.gD * ((-1. * self.counts[k] / 2) * self._cache_log_pi + self._cache_gammaln_by_2[v_N] - self._cache_gammaln_by_2[self.prior.v_0] + (1. / 2) * (np.log(self.prior.k_0) - np.log(k_N))) + np.sum((self.prior.v_0 / 2) * np.log(self.prior.S_0) - (v_N / 2) * np.log(S_N))

        return log_post_Z
    
    def get_posterior_probability_Z_k_cat(self, k):
        """
        Compute the posterior probability of Z_k for categorical features.

        Args:
            k (int): Index of the cluster.

        Returns:
            float: Log posterior probability of Z_k for categorical features.
        """
        return self.cD * (gammaln(self.gamma) - gammaln(self.counts[k] + self.gamma)) + np.sum(np.sum(gammaln(self.catCounts[k] + self.gamma) - gammaln(self.gamma), axis=0))
    
    def get_posterior_probability_Z_k(self, k):
        """
        Compute the posterior probability of Z_k for both Gaussian and categorical features.

        Args:
            k (int): Index of the cluster.

        Returns:
            float: Log posterior probability of Z_k.
        """
        if k >= self.K:
            return gammaln(self.alpha / self.K_max)
        else:
            return self.get_posterior_probability_Z_k_gauss(k) + self.get_posterior_probability_Z_k_cat(k) + gammaln(self.alpha / self.K_max + self.counts[k])
    
    def _students_t_prior(self, i):
        """
        Compute the log probability of data point i under the Student's t prior.

        Args:
            i (int): Index of the data point.

        Returns:
            float: Log probability of data point i under the Student's t prior.
        """
        mu = self.prior.m_0
        inv_var = self._cache_inv_var_prior
        v = self.prior.v_0

        delta = self.X[i, :] - mu

        return self._cache_post_pred_coeff_prior - ((v + 1.) / 2. * (np.log(1. + 1. / v * np.square(delta) * inv_var)).sum())




# class for statistics of the Gassian clusters (diagonal covariance matrix) with feature selection incorporated
class gaussianClustersDiagFS(object):
    """
    Class for Gaussian clusters with diagonal covariance matrices and feature selection.

    Attributes:
        X (numpy.ndarray): Input data matrix of shape (N, D).
        N (int): Number of data points.
        D (int): Number of features.
        prior (PriorParameters): Prior parameters for Gaussian features.
        K_max (int): Maximum number of clusters including empty ones.
        alpha (float): Concentration parameter for the Dirichlet process.
        assignments (numpy.ndarray): Initial assignments of data points to clusters.
        FS (bool): Flag indicating if feature selection is enabled.
        features (numpy.ndarray): Matrix specifying feature selection for each cluster.
        sum_X (numpy.ndarray): Sum of data points for each cluster.
        square_prod_X (numpy.ndarray): Sum of squared data points for each cluster.
        counts (numpy.ndarray): Number of data points assigned to each cluster.
        log_det_covariances (numpy.ndarray): Log determinant of covariance matrix for each cluster.
        log_det_covariances_all (numpy.ndarray): Log determinant of covariance matrix for each cluster (all features).
        inv_covariances (numpy.ndarray): Inverse covariance matrix for each cluster.
    """

    def __init__(self, X, prior, alpha, K, assignments, FS, features):
        """
        Initialize the Gaussian clusters model with feature selection.

        Args:
            X (numpy.ndarray): Input data matrix of shape (N, D).
            prior (PriorParameters): Prior parameters for Gaussian distributions.
            alpha (float): Concentration parameter for the Dirichlet process.
            K (int): Maximum number of clusters including empty ones.
            assignments (numpy.ndarray): Initial assignments of data points to clusters.
            FS (bool): Flag indicating if feature selection is enabled.
            features (numpy.ndarray): Matrix specifying feature selection for each cluster.
        """
        self.X = X
        self.N, self.D = X.shape
        self.prior = prior
        self.K_max = K
        self.alpha = alpha

        if FS:
            if len(features) == 0:
                features_imp = np.random.randint(0, 2, (self.K_max, self.D))
            else:
                features_imp = features
        else:
            features_imp = np.ones((self.K_max, self.D), int)

        self.features = features_imp

        # Initialize cluster statistics
        self.sum_X = np.zeros((self.K_max, self.D), float)
        self.square_prod_X = np.zeros((self.K_max, self.D), float)
        self.counts = np.zeros(self.K_max, int)

        # Initialize background statistics for feature selection
        self.background_sum_X = np.zeros((self.D), float)
        self.background_square_prod_X = np.zeros((self.D), float)
        self.background_counts = 0

        # Initialize hyperparameters' attributes
        self.log_det_covariances = np.zeros(self.K_max)
        self.log_det_covariances_all = np.zeros((self.K_max, self.D))
        self.inv_covariances = np.zeros((self.K_max, self.D))

        # Cache values for computation efficiency
        self._cache()

        # Initialize cluster count
        self.K = 0

        # Assign initial cluster assignments
        self.assignments = assignments
        
        # Add assigned clusters
        for k in range(self.assignments.max() + 1):
            for i in np.where(self.assignments == k)[0]:
                self.add_assignment(i, k)

    def _cache(self):
        """
        Precompute/cache values for efficient computation.
        """
        self._cache_square_X = np.square(self.X)
        self._cache_prior_square_m_0 = np.square(self.prior.m_0)
        Ns = np.concatenate([[1], np.arange(1, self.prior.v_0 + 2 * self.D * self.N + 4)])
        self._cache_gammaln_by_2 = gammaln(Ns / 2.)
        self._cache_log_pi = np.log(np.pi)
        self._cache_log_Vs = np.log(Ns)
        self._cache_gammaln_alpha = gammaln(self.alpha)
        self._cache_prod_k0m0m0 = self.prior.k_0 * self.prior.m_0 * self.prior.m_0
        self._cache_prod_k0m0 = self.prior.k_0 * self.prior.m_0
        self._cache_partial_S_sum = self.prior.S_0 + self._cache_prod_k0m0m0
        var = self.prior.S_0 * (self.prior.k_0 + 1.) / (self.prior.k_0 * self.prior.v_0)
        self._cache_inv_var_prior = 1. / var
        self._cache_log_var_prod_prior = 0.5 * np.log(var).sum()
        self._cache_post_pred_coeff_prior = self.D * (
            self._cache_gammaln_by_2[self.prior.v_0 + 1] - self._cache_gammaln_by_2[self.prior.v_0]
            - 0.5 * self._cache_log_Vs[self.prior.v_0] - 0.5 * self._cache_log_pi
        )

    def cache_cluster_stats(self, k):
        """
        Cache cluster statistics for later use.

        Args:
            k (int): Cluster index.

        Returns:
            tuple: Cached cluster statistics including log determinants, inverses, counts, sum, and square products.
        """
        return (
            self.log_det_covariances[k].copy(),
            self.inv_covariances[k].copy(),
            self.counts[k].copy(),
            self.sum_X[k].copy(),
            self.square_prod_X[k].copy()
        )

    def restore_cluster_stats(self, k, log_det_covariance, inv_covariance, count, sum_X, outer_prod_X):
        """
        Restore cluster statistics from cached values.

        Args:
            k (int): Cluster index.
            log_det_covariance (float): Log determinant of covariance matrix.
            inv_covariance (numpy.ndarray): Inverse covariance matrix.
            count (int): Number of data points in the cluster.
            sum_X (numpy.ndarray): Sum of data points in the cluster.
            outer_prod_X (numpy.ndarray): Outer product of data points in the cluster.
        """
        self.log_det_covariances[k] = log_det_covariance
        self.inv_covariances[k, :] = inv_covariance
        self.counts[k] = count
        self.sum_X[k] = sum_X
        self.square_prod_X[k] = outer_prod_X


    def add_assignment(self, i, k):
        """
        Add a data point to a cluster.

        Args:
            i (int): Index of the data point.
            k (int): Index of the cluster.
        """
        # assigning new cluster k for the ith observation
        if k == self.K:
            self.K += 1
            
            # initializing the partial attributes for new k
            self.sum_X[k, :] = np.zeros(self.D)
            self.square_prod_X[k, :] = np.zeros(self.D)

        self.assignments[i] = k

        # updating the partial hyperparameters
        self.sum_X[k, :] += self.X[i]
        self.square_prod_X[k, :] += self._cache_square_X[i]
        self.counts[k] += 1

        # updating covariance matrix attributes
        self._update_log_det_covariance_and_inv_covariance(k) 


    def del_assignment(self, i):
        """
        Remove a data point from its assigned cluster.

        Args:
            i (int): Index of the data point.
        """
        # delete the assignment and attributes of i-th data vector
        k = self.assignments[i]

        if k != -1 :
            self.assignments[i] = -1
            self.counts[k] -= 1
            if self.counts[k] == 0:
                
                # if cluster is empty, remove it
                self.empty_cluster(k)
            else:

                # update attributions
                self.sum_X[k, :] -= self.X[i]
                self.square_prod_X[k, :] -= self._cache_square_X[i]
                
                self._update_log_det_covariance_and_inv_covariance(k)


    def empty_cluster(self, k):
        """
        Remove an empty cluster and update the cluster statistics.

        Args:
            k (int): Index of the empty cluster.
        """
        self.K -= 1
        if k != self.K:

            # put all stats from last cluster into the empty cluster (one which is being remopved)
            self.sum_X[k, :] = self.sum_X[self.K, :]
            self.square_prod_X[k, :] = self.square_prod_X[self.K, :]
            self.counts[k] = self.counts[self.K]
            self.log_det_covariances[k] = self.log_det_covariances[self.K]
            self.inv_covariances[k, :] = self.inv_covariances[self.K, :]
            self.assignments[np.where(self.assignments == self.K)] = k
            # self.features[k], self.features[self.K] = self.features[self.K],self.features[k]

        # # empty out stats from last cluster
        # self.log_det_covariances[self.K] = 0.
        # self.inv_covariances[self.K, :, :].fill(0.)
        # self.counts[self.K] = 0

        # fill out priors stats from last cluster
        self.counts[self.K] = 0
        self._update_log_det_covariance_and_inv_covariance_priors(self.K)

        self.sum_X[self.K, :] = np.zeros(self.D)
        self.square_prod_X[self.K, :] = np.zeros(self.D)


    def get_background_params(self):
        """
        Calculate background parameters for feature importance.

        Returns:
            tuple: Background parameters (k_N_bg, v_N_bg, m_N_bg, S_sum).
        """
        neg_feature_imp = 1 - self.features
        background_sum_X = neg_feature_imp * self.sum_X
        background_sum_square_X = neg_feature_imp * self.square_prod_X

        background_counts = neg_feature_imp * self.counts[:, np.newaxis]

        k_N_bg = self.prior.k_0 + background_counts.sum(axis=0)
        v_N_bg = self.prior.v_0 + background_counts.sum(axis=0)

        m_N_bg = (background_sum_X.sum(axis = 0) + self._cache_prod_k0m0)/k_N_bg

        S_sum = self._cache_partial_S_sum + background_sum_square_X.sum(axis = 0)

        return k_N_bg, v_N_bg, m_N_bg, S_sum
    
    
    def log_post_pred(self, i):
        """
        Calculate the log posterior predictive probability for a data point.

        Args:
            i (int): Index of the data point.

        Returns:
            numpy.ndarray: Log posterior predictive probabilities for each cluster.
        """
        k_Ns = self.prior.k_0 + self.counts[:self.K_max]
        v_Ns = self.prior.v_0 + self.counts[:self.K_max]
        m_Ns = (self.sum_X[:self.K_max] + self._cache_prod_k0m0)/k_Ns[:, np.newaxis]

        deltas = m_Ns - self.X[i]
        
        res = np.zeros(self.K_max)
        res[:self.K_max] =  self.D * (
                self._cache_gammaln_by_2[v_Ns + 1] - self._cache_gammaln_by_2[v_Ns] 
                - 0.5*self._cache_log_Vs[v_Ns] - 0.5*self._cache_log_pi
                ) - 0.5*self.log_det_covariances[:self.K_max] - (v_Ns + 1)/2. * (np.log(
                1 + np.square(deltas)*self.inv_covariances[:self.K_max]*(1./v_Ns[:, np.newaxis])
                )).sum(axis=1)

        return res


    def log_post_pred0(self, i):
        """
        Calculate the log posterior predictive probability for a data point, considering feature importance.

        Args:
            i (int): Index of the data point.

        Returns:
            numpy.ndarray: Log posterior predictive probabilities for each cluster and feature.
        """
        ans = np.zeros((self.K_max, self.D))
        k_N_bg, v_N_bg, m_N_bg, S_sum = self.get_background_params()

        for k in range(self.K_max):
            k_Ns, v_Ns, m_Ns, S_Ns = self.get_post_hyperparameters(k)
            for j in range(self.D):
                if self.features[k][j] == 1:

                    delta = m_Ns[j] - self.X[i][j]
                    var = (1 + k_Ns)*(S_Ns[j])/(k_Ns*v_Ns)
                    ans[k][j] = self._cache_gammaln_by_2[v_Ns + 1] - self._cache_gammaln_by_2[v_Ns] -0.5*(self._cache_log_pi + np.log(v_Ns) + np.log(var)) - 0.5*(v_Ns + 1) * np.log(1 + np.square(delta)/(v_Ns*var))

                else:
                    k_N, v_N, m_N, S_N_sum = k_N_bg[j], v_N_bg[j], m_N_bg[j], S_sum[j]
                    S_N = S_N_sum - k_N*np.square(m_N)
                    delta = m_N - self.X[i][j]
                    var = (1 + k_N)*(S_N)/(k_N*v_N)

                    ans[k][j] = self._cache_gammaln_by_2[v_N + 1] - self._cache_gammaln_by_2[v_N] -0.5*(self._cache_log_pi + np.log(v_N) + np.log(var)) - 0.5*(v_N + 1) * np.log(1 + np.square(delta)/(v_N*var))

        return ans.sum(axis=1)
    

    def _update_log_det_covariance_and_inv_covariance_priors(self, k):
        """
        Update log determinant of covariance and inverse covariance for priors.

        Args:
            k (int): Index of the cluster.
        """
        self.log_det_covariances[k] = self._cache_log_var_prod_prior
        self.inv_covariances[k, :] = self._cache_inv_var_prior


    def _update_log_det_covariance_and_inv_covariance(self, k):
        """
        Update log determinant of covariance and inverse covariance for a cluster.

        Args:
            k (int): Index of the cluster.
        """
        # update the log_det_covariance and inv_covariance for cluster k       
        k_N = self.prior.k_0 + self.counts[k] 
        v_N = self.prior.v_0 + self.counts[k] 
        m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N 

        # constructing covariance matrix, (S_N = S_N_partials - k_N*m_M*m_N')
        var = (k_N + 1)/(k_N*v_N) * (self._cache_partial_S_sum + self.square_prod_X[k] - k_N * np.square(m_N))
        self.log_det_covariances[k] = np.log(var).sum()
        self.log_det_covariances_all[k] = np.log(var)

        #inverse of covariance matrix
        self.inv_covariances[k, :] = 1./var


    def get_post_hyperparameters(self, k):
        """
        Get posterior hyperparameters for a cluster.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Posterior hyperparameters (k_N, v_N, m_N, S_N).
        """
        # return posterior hyperparameters
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N
        S_N = self._cache_partial_S_sum + self.square_prod_X[k] - k_N*np.square(m_N)
        return k_N, v_N, m_N, S_N
    

    def get_post_posterior_hyperparameters(self, k):
        """
        Get post posterior hyperparameters for a cluster, including posterior updates.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Posterior hyperparameters (post_k_N, post_v_N, post_m_N, post_S_N).
        """
        k_N = self.prior.k_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N

        post_k_N = k_N + self.counts[k]
        post_v_N = self.prior.v_0 + 2*self.counts[k]

        # post_m_N = ((k_N + 1)*self.sum_X[k] + self.prior.k_0*self.prior.v_0)/(k_N*post_k_N)
        post_m_N = (self.sum_X[k] + m_N*k_N)/post_k_N
        post_S_N = self.prior.S_0 + self._cache_prior_square_m_0 + 2*self.square_prod_X[k] - post_k_N*np.square(post_m_N) 
        return post_k_N, post_v_N, post_m_N, post_S_N


    def get_posterior_probability_Z_k(self, k):
        """
        Calculate the posterior probability of Z_k.

        Args:
            k (int): Index of the cluster.

        Returns:
            float: Log posterior probability of Z_k.
        """
        if k >= self.K:
            return gammaln(self.alpha/self.K_max)
        else:
            k_N, v_N, m_N, S_N = self.get_post_hyperparameters(k)
            
            post_k_N, post_v_N, post_m_N, post_S_N = self.get_post_posterior_hyperparameters(k)
            log_post_Z = self.D * ((-1.*self.counts[k]/2)*self._cache_log_pi + self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N] + (1./2)*(np.log(k_N) - np.log(post_k_N))) + np.sum((v_N/2)*np.log(S_N) - (post_v_N/2)*np.log(post_S_N)) +  gammaln(self.alpha/self.K_max + self.counts[k])

            # log_post_Z = self.D * ((-1.*self.counts[k]/2)*self._cache_log_pi + self._cache_gammaln_by_2[v_N] - self._cache_gammaln_by_2[self.prior.v_0] + (1./2)*(np.log(self.prior.k_0) - np.log(k_N))) + np.sum((self.prior.v_0/2)*np.log(self.prior.S_0) - (v_N/2)*np.log(S_N)) +  gammaln(self.alpha/self.K_max + self.counts[k])

            return log_post_Z
    

    def sample_cluster_params(self, k):    
        """
        Sample cluster parameters from the posterior NIW distribution.

        Args:
            k (int): Index of the cluster.

        Returns:
            tuple: Sampled mean vector and covariance matrix.
        """
        # get the attributions first
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N
        S_N = self._cache_partial_S_sum + self.square_prod_X[k] - k_N*np.square( m_N)

        # marginal of sigma
        sigma = invgamma.rvs(v_N, scale=S_N)

        mu = np.zeros(self.D)

        # marginal of mu
        for j in range(self.D):
            mu[j] = np.random.normal(m_N[j], sigma[j]/k_N)
        
        return mu, sigma


    def get_likelihood_bg(self, k, j):
        """
        Calculate the likelihood of background for a specific cluster and feature.

        Args:
            k (int): Index of the cluster.
            j (int): Index of the feature.

        Returns:
            float: Log likelihood of background.
        """
        data_i = np.where(self.assignments == k)[0]
        k_N_bg, v_N_bg, m_N_bg, S_sum = self.get_background_params()
        
        ans = 0
        for i in data_i:
            delta = m_N_bg - self.X[i][j]
            var  = (k_N_bg + 1)/(k_N_bg*v_N_bg) * (S_sum - k_N_bg * np.square(m_N_bg))
            ans += self._cache_gammaln_by_2[v_N_bg] - self._cache_gammaln_by_2[v_N_bg] - 0.5*self._cache_log_Vs[v_N_bg] - 0.5*self._cache_log_pi - 0.5*np.log(var) - ((v_N_bg + 1)/2.) * np.log(1 + np.square(delta)*(1./var)*(1./v_N_bg))
    
        return ans[j]


    def log_prob_unimp_likelihood(self):
        """
        Calculate the log probability of likelihood for all unimportant features.
        """
        
        k_N_bg, v_N_bg, m_N_bg, S_sum = self.get_background_params()
        ans = np.zeros(self.D)

        for j in range(self.D):
            k_N, v_N, m_N, S_N = k_N_bg[j], v_N_bg[j], m_N_bg[j], S_sum[j]
            for i in range(self.N):
                delta = m_N - self.X[i][j]
                var = (1 + k_N)*(S_N - k_N * np.square(m_N))/(k_N*v_N)

                ans[j] += self._cache_gammaln_by_2[v_N + 1] - self._cache_gammaln_by_2[v_N] -0.5*(self._cache_log_pi + v_N + var) - 1/2*(v_N + 1) * np.log(1 + np.square(delta)/(v_N*var))

        return ans                    

    def log_prob_unimp_marginal(self, lamb):
        """
        Calculate the log probability of the marginal distribution for unimportant features.

        Args:
            lamb (float): Regularization parameter for feature selection.

        Returns:
            numpy.ndarray: Log probability of marginal distribution for unimportant features.
        """
        ans = np.zeros((self.K_max, self.D))
        for j in range(self.D):

            # for k in range(self.K_max - 1, -1, -1):
            for k in range(self.K_max):
                # Copy feature importance and set current feature as unimportant
                Ks = self.features[:, j].copy()
                Ks[k] = 0
                Ks_required = 1 - Ks

                # Calculate statistics for unimportant features
                counts_unimp = Ks_required * self.counts
                sum_unimp = Ks_required * self.sum_X[:, j]
                prod_unimp = Ks_required * self.square_prod_X[:, j]
                # print(self.features)
                # print(k, counts_unimp.sum(), sum_unimp.sum(), prod_unimp.sum())

                # Calculate posterior hyperparameters
                k_N = self.prior.k_0 + counts_unimp.sum()
                v_N = self.prior.v_0 + counts_unimp.sum()
                m_N = (sum_unimp.sum() + self._cache_prod_k0m0m0[j]) / k_N
                S_N = self._cache_partial_S_sum[j] + prod_unimp.sum() - k_N * np.square(m_N)

                post_k_N = k_N + counts_unimp.sum()
                post_v_N = v_N + counts_unimp.sum()
                post_m_N = (sum_unimp.sum() + m_N*k_N)/post_k_N
                post_S_N = S_N + k_N*np.square(m_N) + prod_unimp.sum() - post_k_N * np.square(post_m_N)

                # breakpoint()

                ans[k][j] += self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N] + 0.5*(np.log(k_N) - np.log(post_k_N)) + (v_N / 2)*np.log(S_N) - (post_v_N / 2)*np.log(post_S_N) - 1/2 * counts_unimp.sum() * self._cache_log_pi
                ans[k][j] -= 2*lamb

            # print(ans)
            # breakpoint()

        return ans

    def log_prob_imp_likelihood(self):
        """
        Calculate the log probability of the likelihood for important features.

        Returns:
            numpy.ndarray: Log probability of likelihood for important features.
        """
        ans = np.zeros((self.K_max, self.D))

        # Calculate for existing clusters
        for k in range(self.K):
            k_N, v_N, m_N, S_N = self.get_post_hyperparameters(k)
            for j in range(self.D):
                for i in range(self.N):
                    delta = m_N[j] - self.X[i][j]
                    var = (1 + k_N) * (S_N[j]) / (k_N * v_N)
                    ans[k][j] += (
                        self._cache_gammaln_by_2[v_N + 1] - self._cache_gammaln_by_2[v_N] -
                        0.5 * (self._cache_log_pi + v_N + var) -
                        0.5 * (v_N + 1) * np.log(1 + np.square(delta) / (v_N * var))
                    )

        # Calculate for potential new clusters
        for k in range(self.K, self.K_max):
            for j in range(self.D):
                for i in range(self.N):
                    ans[k][j] += self._students_t_prior(i)

        return ans

    def log_prob_imp_marginal(self, lamb):
        """
        Calculate the log probability of the marginal distribution for important features.

        Args:
            lamb (float): Regularization parameter for feature selection.

        Returns:
            numpy.ndarray: Log probability of marginal distribution for important features.
        """
        ans = np.zeros((self.K_max, self.D))

        # Calculate for all clusters
        for k in range(self.K_max):
            k_N, v_N, m_N, S_N = self.get_post_hyperparameters(k)
            post_k_N, post_v_N, post_m_N, post_S_N = self.get_post_posterior_hyperparameters(k)

            for j in range(self.D):
                ans[k][j] = (
                    self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N] +
                    0.5 * (np.log(k_N) - np.log(post_k_N)) +
                    (v_N / 2) * np.log(S_N[j]) - (post_v_N / 2) * np.log(post_S_N[j]) -
                    0.5 * self.counts[k] * self._cache_log_pi
                )
                ans[k][j] -= 2 * lamb

        # Adjust for unimportant features in other clusters
        for j in range(self.D):
            for k in range(self.K_max):
                Ks = self.features[:, j].copy()
                Ks[k] = 1
                Ks_required = 1 - Ks

                if sum(Ks_required) != 0:
                    counts_unimp = Ks_required * self.counts
                    sum_unimp = Ks_required * self.sum_X[:, j]
                    prod_unimp = Ks_required * self.square_prod_X[:, j]

                    k_N = self.prior.k_0 + counts_unimp.sum()
                    v_N = self.prior.v_0 + counts_unimp.sum()
                    m_N = (sum_unimp.sum() + self._cache_prod_k0m0m0[j]) / k_N
                    S_N = self._cache_partial_S_sum[j] + prod_unimp.sum() - k_N * np.square(m_N)

                    post_k_N = k_N + counts_unimp.sum()
                    post_v_N = v_N + counts_unimp.sum()
                    post_m_N = (sum_unimp.sum() + m_N * k_N) / post_k_N
                    post_S_N = S_N + k_N * np.square(m_N) + prod_unimp.sum() - post_k_N * np.square(post_m_N)

                    ans[k][j] += (
                        self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N] +
                        0.5 * (np.log(k_N) - np.log(post_k_N)) +
                        (v_N / 2) * np.log(S_N) - (post_v_N / 2) * np.log(post_S_N) -
                        0.5 * (k_N - self.prior.k_0) * self._cache_log_pi
                    )
                    ans[k][j] -= 2 * lamb

        return ans

    def map_cluster_params(self, k):
        """
        Calculate the Maximum A Posteriori (MAP) estimates of cluster parameters.

        Args:
            k (int): Cluster index.

        Returns:
            tuple: MAP estimates of cluster's mean (m_N) and covariance (sigma).
        """
        k_N = self.prior.k_0 + self.counts[k]
        v_N = self.prior.v_0 + self.counts[k]
        m_N = (self.sum_X[k] + self._cache_prod_k0m0) / k_N
        sigma = (self._cache_partial_S_sum + self.outer_prod_X[k] - k_N * np.outer(m_N, m_N)) / (v_N + self.D + 2)
        return (m_N, sigma)

    def _students_t_prior(self, i):
        """
        Calculate the log probability of a data point under the Student's t prior.

        Args:
            i (int): Index of the data point.

        Returns:
            float: Log probability under the Student's t prior.
        """
        inv_var = self._cache_inv_var_prior
        v = self.prior.v_0
        mu = self.prior.m_0
        delta = self.X[i, :] - mu

        return self._cache_post_pred_coeff_prior - ((v + 1.) / 2. * (np.log(1. + 1. / v * np.square(delta) * inv_var)))

    def get_no_free_param(self):
        """
        Calculate the number of free parameters in the model.

        Returns:
            int: Number of free parameters.
        """
        ans = 0
        for j in range(self.D):
            ans += self.features[:, j].sum()
            if 0 in self.features[:, j]:
                ans += 1
        return ans


# Class for statistics of categorical clusters with feature selection
class categoricalClustersFS(object):

    def __init__(self, C, alpha, gamma, K, assignments, FS, features=[]):
        """
        Initialize the categorical clusters with feature selection.

        Args:
            C (numpy.ndarray): Input data matrix.
            alpha (float): Concentration parameter for the Dirichlet process.
            gamma (float): Prior parameter for categorical distribution.
            K (int): Maximum number of clusters.
            assignments (numpy.ndarray): Initial cluster assignments.
            FS (bool): Flag for feature selection.
            features (list): Initial feature importance matrix.
        """
        self.N, self.D = C.shape
        self.alpha = alpha
        self.C = C
        self.gamma = gamma
        self.K_max = K

        # Initialize feature importance matrix
        if FS:
            if len(features) == 0:
                features_imp = np.random.randint(0, 2, (self.K_max, self.D))
            else:
                features_imp = features
        else:
            features_imp = np.ones((self.K_max, self.D), int)

        # Calculate number of categories for each feature
        self.Ms = np.zeros(self.D, int)
        for d in range(self.D):
            self.Ms[d] = len(set(C[:, d]))
        
        # Initialize partial hyperparameters
        self.counts = np.zeros(self.K_max, int)
        self.catCounts = np.zeros((self.K_max, self.Ms.max(), self.D), int) 
        
        # Initialize feature selection parameters
        self.features = features_imp

        # Initialize background counts
        self.background_catCounts = np.zeros((self.Ms.max(), self.D), int) 
        self.background_counts = 0
        
        # Cache some values for efficiency
        self._cache()

        # Initialize cluster assignments
        self.K = 0
        if assignments is None:
            self.assignments = -1 * np.ones(self.N, int)
        else:
            self.assignments = assignments
            
            # Add initial assignments to clusters
            for k in range(self.assignments.max() + 1):
                for i in np.where(self.assignments == k)[0]:
                    self.add_assignment(i, k)

    def _cache(self):
        """
        Pre-compute and cache some values for efficiency.
        """
        self._cache_log_pi = np.log(np.pi)
        self._cache_gammaln_alpha = gammaln(self.alpha)

    def cache_cluster_stats(self, k):
        """
        Cache statistics for a specific cluster.

        Args:
            k (int): Cluster index.

        Returns:
            tuple: Cached cluster statistics (counts, categorical counts).
        """
        return (
            self.counts[k].copy(),
            self.catCounts[k].copy()
        )

    def restore_cluster_stats(self, k, count_N, catCount_N):
        """
        Restore cached statistics for a specific cluster.

        Args:
            k (int): Cluster index.
            count_N (int): Cached count for the cluster.
            catCount_N (numpy.ndarray): Cached categorical counts for the cluster.
        """
        self.counts[k] = count_N
        self.catCounts[k] = catCount_N

    def add_assignment(self, i, k):
        """
        Add a data point to a cluster.

        Args:
            i (int): Index of the data point.
            k (int): Index of the cluster.
        """
        if k == self.K:
            self.K += 1
            
        self.assignments[i] = k
        
        # Update partial hyperparameters
        for d in range(self.D):
            self.catCounts[k][self.C[i][d]][d] += 1

        self.counts[k] += 1

    def del_assignment(self, i):
        """
        Remove a data point from its assigned cluster.

        Args:
            i (int): Index of the data point.
        """
        k = self.assignments[i]

        if k != -1:
            self.assignments[i] = -1
            self.counts[k] -= 1
            for d in range(self.D):
                self.catCounts[k][self.C[i][d]][d] -= 1

            if self.counts[k] == 0:
                self.empty_cluster(k)

    def empty_cluster(self, k):
        """
        Remove an empty cluster and adjust cluster indices.

        Args:
            k (int): Index of the empty cluster.
        """
        self.K -= 1
        if k != self.K:
            self.counts[k] = self.counts[self.K]
            self.assignments[np.where(self.assignments == self.K)] = k
            self.catCounts[k, :] = self.catCounts[self.K, :]

        # Reset stats for the last cluster
        self.counts[self.K] = 0
        self.catCounts[self.K, :] = np.zeros((self.Ms.max(), self.D), int)

    def log_post_pred(self, i):
        """
        Compute log posterior predictive probability for a data point.

        Args:
            i (int): Index of the data point.

        Returns:
            numpy.ndarray: Log posterior predictive probabilities for each cluster.
        """
        gamma_Ns = self.gamma + self.catCounts

        res = np.zeros((self.K_max, self.D))
        for d in range(self.D):
            res[:,d] = np.log(gamma_Ns[:, self.C[i][d], d]) - np.log(self.counts + self.gamma*self.Ms[d])

        return res.sum(axis=1)

    def log_post_pred0(self, i):
        """
        Compute log posterior predictive probability for a data point with feature selection.

        Args:
            i (int): Index of the data point.

        Returns:
            numpy.ndarray: Log posterior predictive probabilities for each cluster.
        """
        ans = np.zeros((self.K_max, self.D))
        neg_feature_imp = 1 - self.features

        counts_bg = (neg_feature_imp * self.counts[:, np.newaxis]).sum(axis = 0)
        catCounts_bg = (neg_feature_imp[:, np.newaxis]*self.catCounts).sum(axis=0)
        gamma_bg = self.gamma + catCounts_bg

        for k in range(self.K_max):
            gamma_Ns = self.gamma + self.catCounts[k]

            for j in range(self.D):
                if self.features[k][j] == 1:
                    ans[k][j] = np.log(gamma_Ns[self.C[i][j], j]) - np.log(self.counts[k] + self.gamma*self.Ms[j])
                else:
                    ans[k][j] = np.log(gamma_bg[self.C[i][j], j]) - np.log(counts_bg[j] + self.gamma*self.Ms[j])

        return ans.sum(axis=1)

    def log_prob_unimp_likelihood(self):
        """
        Compute log probability of unimportant features likelihood.

        Returns:
            numpy.ndarray: Log probabilities for each feature.
        """
        gamma_Ns = self.gamma + self.catCounts
        gamma_Ns_j = np.zeros((self.D, self.Ms.max()))
        counts_bg = 0

        for j in range(self.D):
            for k in range(self.K):
                if self.features[k][j] == 0:
                    gamma_Ns_j[j] += gamma_Ns[k, :, j]    
                    counts_bg += self.counts[k]

        gamma_Ns_j = gamma_Ns_j.sum(axis=0)
        ans = np.zeros(self.D)

        for j in range(self.D):
            for i in range(self.N):
                ans[j] += np.log(gamma_Ns_j[self.C[i][j]]) - np.log(counts_bg + self.gamma * self.Ms.max())

        return ans

    def log_prob_unimp_marginal(self, lamb):
        """
        Compute log probability of unimportant features marginal.

        Args:
            lamb (float): Regularization parameter.

        Returns:
            numpy.ndarray: Log probabilities for each cluster and feature.
        """
        ans = np.zeros((self.K_max, self.D))
        for j in range(self.D):
            for k in range(self.K_max):
                Ks = self.features[:, j]
                Ks[k] = 0
                Ks_required = 1 - Ks

                catCounts_unimp = (Ks_required[:, np.newaxis]*self.catCounts[:,:, j]).sum(axis = 0)
                counts_unimp = (Ks_required * self.counts).sum()

                gamma_N = self.gamma + catCounts_unimp
                post_gamma_N = gamma_N + catCounts_unimp

                ans[k][j] += gammaln(gamma_N.sum()) + gammaln(post_gamma_N).sum() - gammaln(gamma_N).sum() - gammaln(counts_unimp + gamma_N.sum())
                ans[k][j] -= 2*lamb

        return ans

    def log_prob_imp_marginal(self, lamb):
        """
        Compute log probability of important features marginal.

        Args:
            lamb (float): Regularization parameter.

        Returns:
            numpy.ndarray: Log probabilities for each cluster and feature.
        """
        ans = np.zeros((self.K_max, self.D))
        for k in range(self.K_max):
            gamma_N = self.gamma + self.catCounts[k]
            post_gamma_N = gamma_N + self.catCounts[k]             

            for j in range(self.D):
                ans[k][j] = gammaln(gamma_N[:, j].sum()) + gammaln(post_gamma_N[:, j]).sum() - gammaln(gamma_N[:, j]).sum() - gammaln(self.counts[k] + gamma_N[:, j].sum())
                ans[k][j] -= 2*lamb

        for j in range(self.D):
            for k in range(self.K_max):
                Ks = self.features[:, j]
                Ks[k] = 1
                Ks_required = 1 - Ks

                if sum(Ks) != 0:
                    catCounts_unimp = (Ks_required[:, np.newaxis]*self.catCounts[:,:, j]).sum(axis = 0)
                    counts_unimp = (Ks_required * self.counts).sum()

                    gamma_N = self.gamma + catCounts_unimp
                    post_gamma_N = gamma_N + catCounts_unimp

                    ans[k][j] += gammaln(gamma_N.sum()) + gammaln(post_gamma_N).sum() - gammaln(gamma_N).sum() - gammaln(counts_unimp + gamma_N.sum())
                    ans[k][j] -= 2*lamb
            
        return ans

    def log_prob_imp_likelihood(self):
        """
        Compute log probability of important features likelihood.

        Returns:
            numpy.ndarray: Log probabilities for each cluster and feature.
        """
        gamma_Ns = self.gamma + self.catCounts[:self.K]
        ans = np.zeros((self.K, self.D))

        for j in range(self.D):
            for k in range(self.K):
                for i in range(self.N):
                    ans[k, j] += np.log(gamma_Ns[k, self.C[i][j], j]) - np.log(self.counts[k] + self.gamma * self.Ms.max())

        return ans

    def get_posterior_probability_Z_k(self, k):
        """
        Compute the posterior probability of Z_k.

        Args:
            k (int): Cluster index.

        Returns:
            float: Log posterior probability of Z_k.
        """
        gamma_N = self.gamma + self.catCounts[k]
        post_gamma_N = gamma_N + self.catCounts[k]
        
        log_post_Z = np.zeros((self.D))
        for d in range(self.D):
            log_post_Z[d] = gammaln(gamma_N[:, d].sum()) + gammaln(post_gamma_N[:, d]).sum() - gammaln(gamma_N[:, d]).sum() - gammaln(self.counts[k] + gamma_N[:, d].sum())

        return log_post_Z.sum() + gammaln(self.alpha/self.K_max + self.counts[k])

    # The following code is commented out and may be obsolete or for future use
    """
    if k >= self.K:
        return gammaln(self.alpha/self.K_max) 
    
    else:
        gamma_N = self.gamma + self.catCounts[k]
        log_post_Z = np.zeros((self.D))
        for d in range(self.D):
            # log_post_Z[d] = gammaln(self.gamma * self.Ms[d]) +  gammaln(gamma_N[:, d]).sum() - self.Ms[d] * gammaln(self.gamma) -  gammaln(self.gamma * self.Ms[d] + self.counts[k])
            log_post_Z[d] = gammaln(gamma_N[:, d]).sum() -  gammaln(self.gamma * self.Ms[d] + self.counts[k])

        return log_post_Z.sum()
    """



########### T R A S H ##################### T R A S H ############# T R A S H ################ T R A S H ############### T R A S H ############ T R A S H ############# T R A S H ############ T R A S H ############ T R A S H ################      
        
# class poissonClustersDiag(object):

#     def __init__(self, X, prior, alpha, K, assignments=None):
#         # assignments is initial assignments of clusters
#         # K-max is the maximum number of clusters including the empty ones

#         self.X = X
#         self.N, self.D = X.shape
#         self.prior = prior
#         self.K_max = K
#         self.alpha = alpha
        
#         ####### partial_hyperparaneters_attr #############
#         self.sum_X = np.zeros((self.K_max, self.D), float)
#         self.square_prod_X = np.zeros((self.K_max, self.D), float)
#         self.counts = np.zeros(self.K_max, int)

#         ####### hyper-parameters' attributes initialization ########

#         # log of determinant of multivariate Student's t distribution associated with each of the K cluster (kx1 vector)
#         self.log_det_covariances = np.zeros(self.K_max)
        
#         # inverse of S_N_partials (Kx(DxD) matrix)
#         self.inv_covariances = np.zeros((self.K_max, self.D))

#         # to avoid recomputing we will cache some log and log gamma values
#         self._cache()

#         # Initialization
#         self.K = 0

#         # assign the initial assignments
#         self.assignments = assignments
        
#         # adding the assigned clusters
#         for k in range(self.assignments.max() + 1):
#             for i in np.where(self.assignments == k)[0]:
#                 self.add_assignment(i, k)


#     def _cache(self):

#         # pre-computing outer products
#         # self._cache_square_X = np.zeros((self.N, self.D))
#         self._cache_square_X = np.square(self.X) 
        
#         self._cache_prior_square_m_0 = np.square(self.prior.m_0)

#         # pre-computing gamma values of possible numbers (for computing student's t)
#         Ns = np.concatenate([[1], np.arange(1, self.prior.v_0 + 2* self.N + 4)])
#         self._cache_gammaln_by_2 = gammaln(Ns/2.)
#         self._cache_log_pi = np.log(np.pi)
#         self._cache_log_Vs = np.log(Ns)
#         self._cache_gammaln_alpha = gammaln(self.alpha)
#         self._cache_prod_k0m0 = self.prior.k_0 * self.prior.m_0
#         self._cache_partial_S_sum = self.prior.S_0 + self.prior.k_0*np.square(self.prior.m_0)

#         var = (self.prior.k_0 + 1.) / (self.prior.k_0*self.prior.v_0) * self.prior.S_0
#         self._cache_inv_var_prior = 1./var
#         self._cache_log_var_prod_prior = 0.5*np.log(var).sum()
#         self._cache_post_pred_coeff_prior = self.D * ( self._cache_gammaln_by_2[self.prior.v_0 + 1] - self._cache_gammaln_by_2[self.prior.v_0]
#                 - 0.5*self._cache_log_Vs[self.prior.v_0] - 0.5*self._cache_log_pi
#                 ) - self._cache_log_var_prod_prior

#     def cache_cluster_stats(self, k):

#         # caching cluster k's statistics in a tuple
#         return (
#             self.log_det_covariances[k].copy(),
#             self.inv_covariances[k].copy(),
#             self.counts[k].copy(),
#             self.sum_X[k].copy(),
#             self.square_prod_X[k].copy()
#             )


#     def restore_cluster_stats(self, k, log_det_covariance, inv_covariance, count, sum_X, outer_prod_X):
    
#         # restore the cluster stats for the attributes
#         self.log_det_covariances[k] = log_det_covariance
#         self.inv_covariances[k, :] = inv_covariance
#         self.counts[k] = count
#         self.sum_X[k] = sum_X
#         self.square_prod_X[k] = outer_prod_X


#     def add_assignment(self, i, k):

#         # assigning new cluster k for the ith observation
#         if k == self.K:
#             self.K += 1
            
#             # initializing the partial attributes for new k
#             self.sum_X[k, :] = np.zeros(self.D)
#             self.square_prod_X[k, :] = np.zeros(self.D)


#         self.assignments[i] = k

#         # updating the partial hyperparameters
#         self.sum_X[k, :] += self.X[i]
#         self.square_prod_X[k, :] += self._cache_square_X[i]
#         self.counts[k] += 1

#         # updating covariance matrix attributes
#         self._update_log_det_covariance_and_inv_covariance(k) 


#     def del_assignment(self, i):

#         # delete the assignment and attributes of i-th data vector
#         k = self.assignments[i]

#         if k != -1 :
#             self.assignments[i] = -1
#             self.counts[k] -= 1
#             if self.counts[k] == 0:
                
#                 # if cluster is empty, remove it
#                 self.empty_cluster(k)
#             else:

#                 # update attributions
#                 self.sum_X[k, :] -= self.X[i]
#                 self.square_prod_X[k, :] -= self._cache_square_X[i]
                
#                 self._update_log_det_covariance_and_inv_covariance(k)


#     def empty_cluster(self, k):
#         self.K -= 1
#         if k != self.K:

#             # put all stats from last cluster into the empty cluster (one which is being remopved)
#             self.sum_X[k, :] = self.sum_X[self.K, :]
#             self.square_prod_X[k, :] = self.square_prod_X[self.K, :]
#             self.counts[k] = self.counts[self.K]
#             self.log_det_covariances[k] = self.log_det_covariances[self.K]
#             self.inv_covariances[k, :] = self.inv_covariances[self.K, :]
#             self.assignments[np.where(self.assignments == self.K)] = k

#         # # empty out stats from last cluster
#         # self.log_det_covariances[self.K] = 0.
#         # self.inv_covariances[self.K, :, :].fill(0.)
#         # self.counts[self.K] = 0

#         # fill out priors stats from last cluster
#         self.counts[self.K] = 0
#         self._update_log_det_covariance_and_inv_covariance_priors(self.K)

#         self.sum_X[self.K, :] = np.zeros(self.D)
#         self.square_prod_X[self.K, :] = np.zeros(self.D)



#     def log_prior(self, i):

#         # probability of x_i under prior alone
#         return self._multivariate_students_t_prior(i)

    
#     def log_post_pred(self, i):

#         # for j in range(self.K_max):
#         #     if self.counts[0] != self.assignments.tolist().count(0):
#         #         print("fucked up")

#         # returns k dimension vector student's t pdf
#         k_Ns = self.prior.k_0 + self.counts[:self.K]
#         v_Ns = self.prior.v_0 + self.counts[:self.K]
#         m_Ns = (self.sum_X[:self.K] + self._cache_prod_k0m0)/k_Ns[:, np.newaxis]

#         deltas = m_Ns - self.X[i]
        
#         res = np.zeros(self.K_max)
#         res[:self.K] =  self.D * (
#                 self._cache_gammaln_by_2[v_Ns + 1] - self._cache_gammaln_by_2[v_Ns] 
#                 - 0.5*self._cache_log_Vs[v_Ns] - 0.5*self._cache_log_pi
#                 ) - 0.5*self.log_det_covariances[:self.K] - (v_Ns + 1)/2. * (np.log(
#                 1 + np.square(deltas)*self.inv_covariances[:self.K]*(1./v_Ns[:, np.newaxis])
#                 )).sum(axis=1)

#         res[self.K:] = self._students_t_prior(i)

#         return res


#     def _update_log_det_covariance_and_inv_covariance_priors(self, k):
#         self.log_det_covariances[k] = self._cache_log_var_prod_prior
#         self.inv_covariances[k, :] = self._cache_inv_var_prior


#     def _update_log_det_covariance_and_inv_covariance(self, k):
    
#         # update the log_det_covariance and inv_covariance for cluster k       
#         k_N = self.prior.k_0 + self.counts[k] 
#         v_N = self.prior.v_0 + self.counts[k] 
#         m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N 

#         # constructing covariance matrix, (S_N = S_N_partials - k_N*m_M*m_N')
#         var = (k_N + 1)/(k_N*v_N) * (self._cache_partial_S_sum + self.square_prod_X[k] - k_N * np.square(m_N))

#         self.log_det_covariances[k] = np.log(var).sum()

#         #inverse of covariance matrix
#         self.inv_covariances[k, :] = 1./var


#     def get_post_hyperparameters(self, k):

#         # return posterior hyperparameters
#         k_N = self.prior.k_0 + self.counts[k]
#         v_N = self.prior.v_0 + self.counts[k]
#         m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N
#         S_N = self._cache_partial_S_sum + self.square_prod_X[k] - k_N*np.square( m_N)
#         return k_N, v_N, m_N, S_N
    
#     def get_post_posterior_hyperparameters(self, k):
#         k_N = self.prior.k_0 + self.counts[k]
#         m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N

#         post_k_N = k_N + self.counts[k]
#         post_v_N = self.prior.v_0 + 2*self.counts[k]

#         # post_m_N = ((k_N + 1)*self.sum_X[k] + self.prior.k_0*self.prior.v_0)/(k_N*post_k_N)
#         post_m_N = (self.sum_X[k] + m_N*k_N)/post_k_N
#         post_S_N = self.prior.S_0 + self._cache_prior_square_m_0 + 2*self.square_prod_X[k] - post_k_N*np.square(post_m_N)
#         # temp = m_N*k_N - self.prior.k_0*self.prior.m_0 - self.sum_X[k]
#         return post_k_N, post_v_N, post_m_N, post_S_N


#     def get_posterior_probability_Z_k(self, k):

#         if k >= self.K:
#             return gammaln(self.alpha/self.K_max)
#         else:
#             k_N, v_N, m_N, S_N = self.get_post_hyperparameters(k)
#             post_k_N, post_v_N, post_m_N, post_S_N = self.get_post_posterior_hyperparameters(k)
#             log_post_Z = self.D * ((-1.*self.counts[k]/2)*self._cache_log_pi + self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N] + (1./2)*(np.log(k_N) - np.log(post_k_N)) + gammaln(self.alpha/self.K_max + self.counts[k])) + ((v_N/2)*np.log(S_N) - (post_v_N/2)*np.log(post_S_N)).sum()
            
#             # k_N, v_N, m_N_scalar, S_N_scalar = self.get_post_hyperparameters(k)            
#             # m_N = np.eye(self.D)*m_N_scalar
#             # S_N = np.eye(self.D)*S_N_scalar
#             # post_k_N, post_v_N, post_m_N_scalar, post_S_N_scalar = self.get_post_posterior_hyperparameters(k)
#             # post_m_N = np.zeros(self.D) + post_m_N_scalar
#             # post_S_N = np.eye(self.D)*post_S_N_scalar
#             # log_post_Z = -1*self.counts[k]*(self.D/2)* self._cache_log_pi + self._cache_gammaln_by_2[post_v_N] - self._cache_gammaln_by_2[v_N] + (v_N/2)*np.log(slogdet(S_N)[1]) - (post_v_N/2)*np.log(slogdet(post_S_N)[1]) - (self.D/2)*(np.log(self.counts[k]) - np.log(2*self.counts[k])) + gammaln(self.alpha/self.K_max + self.counts[k])
            
#             return log_post_Z
    
#     def random_cluster_params(self, k):    

#         # get random mean vector and covariance matrix from the posterior NIW distribution for cluster k
        
#         # get the attributions first
#         k_N = self.prior.k_0 + self.counts[k]
#         v_N = self.prior.v_0 + self.counts[k]
#         m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N
#         S_N = self.S_N_partials[k] - k_N*np.outer(m_N, m_N)

#         # marginal of sigma
#         sigma = invwishart.rvs(df=v_N, scale=S_N)

#         # marginal of mu
#         if self.D == 1:     
#             mu = np.random.normal(m_N, sigma/k_N)
#         else:
#             mu = np.random.multivariate_normal(m_N, sigma/k_N)
#         return mu, sigma
    

#     def map_cluster_params(self, k):
        
#         # MAP estimates of cluster's mu and sigma
#         k_N = self.prior.k_0 + self.counts[k]
#         v_N = self.prior.v_0 + self.counts[k]
#         m_N = (self.sum_X[k] + self._cache_prod_k0m0)/k_N
#         sigma = (self._cache_partial_S_sum + self.outer_prod_X[k] - k_N*np.outer(m_N, m_N))/(v_N + self.D + 2)
#         return (m_N, sigma)
    
    
#     def _students_t_prior(self, i):
#         mu = self.prior.m_0
#         inv_var = self._cache_inv_var_prior
#         v = self.prior.v_0

#         delta = self.X[i, :] - mu

#         return self._cache_post_pred_coeff_prior  - ((v + 1.)/2. * (np.log(1. + 1./v * np.square(delta) * inv_var)).sum())
