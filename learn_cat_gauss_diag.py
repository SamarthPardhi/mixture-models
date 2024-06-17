import numpy as np
from cluster_stats_new import categoricalGaussianClusters
import utils
import time
import argparse
import pickle
import os
import time
from sklearn.metrics.cluster import adjusted_rand_score


class bayesCGMM():
    """
    Bayesian Categorical-Gaussian Mixture Model (GMM) with diagonal covariance matrices.

    This class implements a Gibbs sampler for Bayesian Categorical and Gaussian MM with diagonal covariance matrices.
    It initializes the model with given data, prior, and initial cluster assignments, and provides
    a method to run the Gibbs sampler for a specified number of iterations.
    """

    def __init__(self, X: float, C: int, alpha: float, gamma: int, prior, assignments, isTrueZ=0):

        """
        Initialize the Categorical MM.

        Args:
            X (np.array): 2D NumPy array of shape (n_samples, n_features) containing the Gaussian.
            C (np.array, dtype=float): 2D NumPy array of shape (n_samples, n_features) containing the Categorical data.
            alpha (float): Dirichlet hyperparameter for mixing probabilities, alpha_0.
            prior (utils.NIchi2): Object representing the prior hyperparameters (m_0, k_0, s_0, v_0).
            gamma (float): Dirichlet hyperparameter for catagories
            assignments (np.array): 1D NumPy array of shape (n_samples,) containing initial cluster assignments.        
        """

        if isTrueZ == 1:
            self.trueZ = assignments.copy()
        else:
            self.trueZ = []
        
        self.alpha = alpha

        # Initial total number of clusters
        K = len(set(assignments))
        self.K_max = K
        
        # Total number of samples and categories
        self.N, cD = C.shape

        # Get number of categories for each feature
        self.Ms = np.zeros(cD, int)
        for d in range(cD):
            self.Ms[d] = len(set(C[d]))

        # Setting up the Categorical-Gaussian Cluster object which will track the features and component-wise statistics
        self.clusters = categoricalGaussianClusters(X, C, alpha, prior, gamma, K, assignments)
        self.z_map = assignments
        self.iter_map = 0
        self.log_max_post = -1*np.inf
        self.BIC = 0.
        self.run_id = -1

    def gibbs_sampler(self, n_iter: int, run_id: int, toPrint=True, savePosterior=False, trueAssignments=[]):

        """
        Run the Gibbs sampler for the Bayesian GMM.

        Args:
            n_iter (int): Number of iterations to run the Gibbs sampler.
            run_id (int): Unique identifier for the current run.
            toPrint (bool, optional): If True, print the results for each iteration. Default is True.
            savePosterior (bool, optional): If True, save the posterior score for each data step in each iteration. Default is False.
            trueAssignments (list, optional): Ground truth cluster assignments for calculating Adjusted Rand Index (ARI). Default is an empty list.
        
        """

        self.trueZ = trueAssignments.copy()

        self.run_id = run_id

        # If the posterior is the same for each iteration, a convergence bound can also be set
        posteriorList = []
        ARI_list = []
        
        # If the posterior is the same for each iteration, a convergence bound can also be set
        same_posterior_count = 0
        ass_posterior = 0

        # Log posterior probability
        log_post_Z = np.zeros(self.K_max)
        for k in range(self.K_max):
            log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

        # Print initial information if want to
        if toPrint:
            if len(self.trueZ) != 0:
                print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{round(np.sum(log_post_Z), 3)}, ARI: {round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3)}")
            else:
                print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{round(np.sum(log_post_Z), 3)}")

        # Start the Gibbs sampler
        for i_iter in range(n_iter):
            old_assignments = self.clusters.assignments.copy()
            
            # For each data point
            for i in range(self.clusters.N):

                # Cache the previous cluster statistics if the same cluster is assigned to the current data point
                k_old = self.clusters.assignments[i]
                K_old = self.clusters.K
                stats_old = self.clusters.cache_cluster_stats(k_old)

                # Remove the data point from the data
                self.clusters.del_assignment(i)
                
                # Calculate f(z_i = k | z_[-i], alpha)
                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(self.clusters.N + self.alpha - 1)

                # Calculate f(c_i | C[-i], z_i = k, z_[-i], Gamma, Beta)
                log_prob_x_i = self.clusters.log_post_pred_cat(i) + self.clusters.log_post_pred_gauss(i)
                
                # Get f(z_i = k | z_[-i])
                log_prob_z_k = log_prob_z_k_alpha + log_prob_x_i

                # Sample new cluster identity for the data point using Gumbel-max trick
                k = np.argmax(log_prob_z_k + np.random.gumbel(0, 1, len(log_prob_z_k)))
                # prob_z =  np.exp(log_prob_z_k - logsumexp(log_prob_z_k))
                # k = np.random.choice(len(prob_z), p=prob_z)

                # If an empty cluster is sampled
                if k >= self.clusters.K:
                    k = self.clusters.K

                # If the sampled cluster is the same as the old one and the cluster didn't become empty                                        
                if k==k_old and self.clusters.K == K_old:
                    self.clusters.restore_cluster_stats(k_old, *stats_old)
                    self.clusters.assignments[i] = k_old

                # Assign a new cluster identity
                else:
                    self.clusters.add_assignment(i,k)

                # Save log posterior probability
                if savePosterior:
                    new_ass = self.clusters.assignments.copy()
                    ass_change = old_assignments == new_ass
                    changed_ = []
                    for i in range(self.clusters.N):
                        if not ass_change[i]:
                            changed_.append(old_assignments[i])
                            changed_.append(new_ass[i])
                    changed_ = list(set(changed_))

                    log_post_Z_ = log_post_Z.copy()
                    for j in changed_:
                        log_post_Z_[j] =  self.clusters.get_posterior_probability_Z_k(j)

                    posteriorList.append(np.sum(log_post_Z_))
                    
                    # Calculate the ARI if true assignments are provided
                    if len(self.trueZ) != 0:
                        ARI_list.append(round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3))

            # Get the list of all changed clusters for the iteration
            new_assignments = self.clusters.assignments
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(self.clusters.N):
                if not assignments_change[i]:
                    changed_clusters.append(old_assignments[i])
                    changed_clusters.append(new_assignments[i])
            changed_clusters = list(set(changed_clusters))

            # Get the posterior score
            for j in changed_clusters:
                log_post_Z[j] =  self.clusters.get_posterior_probability_Z_k(j)
            sum_log_post_Z = np.sum(log_post_Z)

            if sum_log_post_Z > self.log_max_post:
                self.log_max_post = sum_log_post_Z
                self.z_map = self.clusters.assignments.copy()
                self.iter_map = i_iter + 1

            if sum_log_post_Z != ass_posterior:
                same_posterior_count = 0
                ass_posterior = sum_log_post_Z
            else:
                same_posterior_count += 1

            if toPrint:
                if len(self.trueZ) != 0:
                    print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{round(sum_log_post_Z, 3)}, ARI: {adjusted_rand_score(self.trueZ, self.clusters.assignments)}")
                else:
                    print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{round(sum_log_post_Z, 3)}")

            if same_posterior_count > 4:
                break

            print(f"{i_iter}/{n_iter}               ",end='\r')

        self.BIC = self.clusters.K*(2*self.clusters.gD) * np.log(self.clusters.N) - (2 * self.log_max_post) 
        self.BIC += self.clusters.K*(np.sum(self.Ms)) * np.log(self.clusters.N) - (2 * self.log_max_post)
        
        if len(self.trueZ) != 0:
            print(f"\nRun: {run_id + 1}, K:{len(set(self.z_map))}, BIC: {self.BIC}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}, ARI(last iter): {round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3)}, ARI(max post): {round(adjusted_rand_score(self.trueZ, self.z_map), 3)}")
        else:
            print(f"\nRun: {run_id + 1}, K:{len(set(self.z_map))}, BIC: {self.BIC}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")
            
                
        postData = {
            "run":run_id,
            "n_iter":n_iter,
            "posterior":posteriorList,
            "ARI":ARI_list
        }

        return postData
        

if __name__ == "__main__":
    model_start_time = time.perf_counter()  # Start timer for model execution

    # Setup argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-fc", required=True, type=argparse.FileType('r'), help="Path to the file containing Gaussian features data")
    parser.add_argument("-fg", required=True, type=argparse.FileType('r'), help="Path to the file containing categorical features data")
    parser.add_argument("-k", required=True, type=int, help="Known number of clusters or maximum number of clusters if unknown")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs with different initial assignments")
    parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the true parameters file (non-pickle file)")
    parser.add_argument("-p", required=False, action="store_true", help="Print results during Gibbs sampling")
    parser.add_argument("-seed", required=False, type=int, help="Set a seed value")

    args = parser.parse_args()  # Parse command-line arguments

    # Initialize global random seed
    global_seed = np.random.randint(1, 2**31 - 1) if args.seed is None else args.seed
    np.random.seed(global_seed)

    ##################################  Extract data ##################################

    X = []  # Initialize list for Gaussian features
    dataFile = args.fg  # Open Gaussian features file
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]

    for line in dataFile:
        X.append(np.array([float(i) for i in line.strip().split(',')]))  # Parse each line into a float array
    X = np.array(X)  # Convert list to numpy array

    C = []  # Initialize list for categorical features
    dataFile = args.fc  # Open categorical features file
    for line in dataFile:
        C.append(np.array([int(float(i)) for i in line.strip().split(',')]))  # Parse each line into an integer array
    C = np.array(C)  # Convert list to numpy array

    N, cD = C.shape  # Number of samples and number of categorical features
    gD = len(X[0])  # Number of Gaussian features

    # Model parameters
    K_max_BIC = args.k  # Maximum number of clusters
    n_iter = 50 if args.i is None else args.i  # Number of iterations for Gibbs sampling

    training_runs = 1 if args.r is None else args.r  # Number of training runs

    # Print initial configuration
    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, gD: {cD}, fD: {gD}, K: {K_max_BIC}, Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set Gaussian hyper-parameters ##################################

    alpha = 1.0  # Dirichlet prior parameter for Gaussian
    m_0 = np.zeros(gD)  # Mean prior
    k_0 = 0.03  # Mean prior scaling
    v_0 = gD + 3  # Degrees of freedom for inverse chi-squared distribution
    S_0 = 0.3 * v_0 * np.ones(gD)  # Scale matrix for inverse chi-squared distribution

    # Create prior object
    prior = utils.NIchi2(m_0, k_0, v_0, S_0)

    ################################## Set Categorical hyper-parameters ##################################

    gamma = 0.2  # Dirichlet prior parameter for categorical data

    ################################## Model ##################################

    print(f"Total training runs: {training_runs}")

    trueFile = args.t  # Path to true parameters file
    if trueFile:
        trueAssignments = np.array([int(line.strip()) for line in trueFile])  # Parse true assignments
        # Initialize model with true assignments
        bayescgmm = bayesCGMM(X, C, alpha, gamma, prior, trueAssignments, 1)
        bayescgmm.gibbs_sampler(n_iter, -1)  # Run Gibbs sampling
    else:
        trueAssignments = []

    max_post = -1 * np.inf  # Initialize max posterior
    least_BIC = -1 * np.inf  # Initialize least BIC
    for i in range(training_runs):
        print(f"\nRun:  {i+1}")

        # Ensure unique initial assignments
        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)

        # Initialize model with random assignments
        bayescgmm = bayesCGMM(X, C, alpha, gamma, prior, starting_assignments)
        bayescgmm.gibbs_sampler(n_iter, i, trueAssignments=trueAssignments)  # Run Gibbs sampling

        # Update best model based on BIC
        if bayescgmm.BIC > least_BIC:
            least_BIC = bayescgmm.BIC
            best_bayescgmm = bayescgmm

    ##################################  Model results ##################################

    z_pred_map = best_bayescgmm.z_map  # Predicted cluster assignments
    predicted_K = len(set(z_pred_map))  # Number of predicted clusters

    # Prepare results for saving
    mu_pred = []
    sigma_pred = []

    preds = {
        "mu": mu_pred,
        "sigma": np.array(sigma_pred),
        "z": z_pred_map,
        "time": time.perf_counter() - model_start_time,
        "z_last_iter": best_bayescgmm.clusters.assignments
    }

    ##################################  Save results ##################################

    outDir = "outCatGauss" if args.o is None else args.o  # Output directory

    if outDir not in os.listdir():
        os.mkdir(outDir)  # Create output directory if it doesn't exist

    outputFileName = f"{dataFilename}"  # Output file name
    outputFilePath = f"{outDir}/{outputFileName}.txt"  # Full path for the output file

    with open(outputFilePath, "w") as wFile:
        # Write model details to file
        wFile.write(f"N: {N}\n")
        wFile.write(f"gD: {gD}\n")
        wFile.write(f"cD: {cD}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {global_seed}\n")
        wFile.write(f"Iterations: {n_iter}\n")
        wFile.write(f"alpha: {alpha}\n")
        wFile.write(f"time: {time.perf_counter() - model_start_time}\n")
        wFile.write(f"BIC score: {best_bayescgmm.BIC}\n")
        wFile.write(f"log max posterior: {best_bayescgmm.log_max_post}\n")
        wFile.write(f"MAP assignments: {best_bayescgmm.z_map}\n")
        wFile.write(f"Last iteration assignments: {best_bayescgmm.clusters.assignments}\n")

        # Write Gaussian hyperparameters
        wFile.write("m_0:")
        np.savetxt(wFile, m_0)
        wFile.write(f"k_0: {k_0}\n")
        wFile.write(f"v_0: {v_0}\n")
        wFile.write("S_0:")
        np.savetxt(wFile, S_0)

    # Save results in a pickle file
    outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

    # Save predicted labels
    outputFile = open(f"{outDir}/{outputFileName}.labels", "wb")
    utils.saveData(outputFile.name, z_pred_map, "labels")

    # Save predicted labels to a file
    with open(f"{outDir}/{outputFileName}.labels", "w") as ff:
        for z_i in z_pred_map:
            ff.write(f"{z_i}\n")

    # Print locations of the saved results
    print(f"The predicted labels are saved in: {outDir}/{outputFileName}.labels")
    print(f"The encoded results are saved in: {outDir}/{outputFileName}.p")
    print(f"The readable results are saved in: {outputFilePath}")