import numpy as np
from cluster_stats_new import gaussianClustersDiagFS
import utils
import time
import argparse
import pickle
import os
import time
from sklearn.metrics.cluster import adjusted_rand_score


class bayesGMM_FS():
    """
    Bayesian Gaussian Mixture Model (GMM) with diagonal covariance matrices also incorporating feature selection.

    This class implements a Gibbs sampler for Bayesian GMM with diagonal covariance matrices.
    It initializes the model with given data, prior, and initial cluster assignments, and provides
    a method to run the Gibbs sampler for a specified number of iterations.
    """

    def __init__(self, X: np.array, prior: utils.NIchi2, alpha: float, assignments: np.array, FS: bool,  features = []):
        """
        Initialize the Bayesian GMM.

        Args:
            X (np.array): 2D NumPy array of shape (n_samples, n_features) containing the data.
            prior (utils.NIchi2): Object representing the prior hyperparameters (m_0, k_0, s_0, v_0).
            alpha (float): Dirichlet hyperparameter for mixing probabilities, alpha_0.
            assignments (np.array): 1D NumPy array of shape (n_samples,) containing initial cluster assignments.
            FS: True if we want to incorporate feature selection else False
            Features: Feature importance matrix
        """

        self.FS = FS
        self.trueZ = []
        self.alpha = alpha

        # Initial total number of clusters
        K = len(set(assignments))
        self.K_max = K
        
        # Setting up the Gaussian Cluster object which will track the features and component-wise statistics
        self.clusters = gaussianClustersDiagFS(X, prior, alpha, K, assignments.copy(), FS, features)
        
        # Initializing the outputs
        self.z_map = assignments
        self.iter_map = 0
        self.log_max_post = -1*np.inf
        self.BIC = 0.
        self.run_id = -1


    def gibbs_sampler(self, n_iter, run_id, toPrint=True, savePosterior=False, trueAssignments=[], greedyRun = False):
        """
        Run the Gibbs sampler for the Bayesian GMM.

        Args:
            n_iter (int): Number of iterations to run the Gibbs sampler.
            run_id (int): Unique identifier for the current run.
            toPrint (bool, optional): If True, print the results for each iteration. Default is True.
            savePosterior (bool, optional): If True, save the posterior score for each data step in each iteration. Default is False.
            trueAssignments (list, optional): Ground truth cluster assignments for calculating Adjusted Rand Index (ARI). Default is an empty list.
            greedyRun (bool, optional): If True, initialize with previous MAP assignments. Default is False.

        Returns:
            dict: A dictionary containing the following keys:
                - 'run': The run_id value.
                - 'n_iter': The n_iter value.
                - 'posterior': A list of posterior probabilities for each iteration (if savePosterior is True).
                - 'ARI': A list of ARI scores for each iteration (if trueAssignments is provided).
        """
        
        if toPrint:
            print(f"Initial features:")
            print(self.clusters.features)

        if len(trueAssignments) > 0:
            self.trueZ = trueAssignments

        self.run_id = run_id
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
                
                # If it's a greedy run, keep track of the posterior probability
                if greedyRun == True:                                
                    old_post_prob = 0
                    for k in range(self.K_max):
                        old_post_prob += self.clusters.get_posterior_probability_Z_k(k)

                # Cache the previous cluster statistics if the same cluster is assigned to the current data point
                k_old = self.clusters.assignments[i]
                K_old = self.clusters.K
                stats_old = self.clusters.cache_cluster_stats(k_old)
                k_counts_old = self.clusters.counts[k_old]

                # Remove the data point from the data
                self.clusters.del_assignment(i)                

                # Calculate f(z_i = k | z_[-i], alpha)
                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(self.clusters.N + self.alpha - 1)

                # Calculate f(x_i | X[-i], z_i = k, z_[-i], Beta)
                log_prob_x_i = self.clusters.log_post_pred0(i)

                # Get f(z_i = k | z_[-i])
                log_prob_z_k = log_prob_z_k_alpha + log_prob_x_i

                # Sample new cluster identity for the data point using Gumbel-max trick
                k_new = np.argmax(log_prob_z_k + np.random.gumbel(0, 1, len(log_prob_z_k)))

                # Track the changed clusters
                changed_ = []

                # If an empty cluster is sampled
                if k_new >= self.clusters.K:
                    k_new = self.clusters.K

                # If the sampled cluster is the same as the old one and the cluster didn't become empty
                if k_new == k_old and self.clusters.K == K_old:
                    self.clusters.restore_cluster_stats(k_old, *stats_old)
                    self.clusters.assignments[i] = k_old

                # Assign a new cluster identity
                else:
                    self.clusters.add_assignment(i, k_new)
                    changed_ += [k_new, k_old]

                if k_counts_old == 1:
                    changed_.append(K_old - 1)
                
                # Check posterior if it's a greedy run
                if greedyRun and len(changed_) > 0:
                    log_post_Z_ = log_post_Z.copy()
                    old_post_i = np.sum(log_post_Z_)
                    for k_i in changed_:
                        log_post_Z_[k_i] =  self.clusters.get_posterior_probability_Z_k(k_i)

                    if old_post_i > np.sum(log_post_Z_):
                        if k_counts_old == 1:
                            self.clusters.add_assignment(i, self.clusters.K)
                        else:    
                            self.clusters.add_assignment(i, k_old)

                # Save log posterior probability
                if savePosterior:

                    if len(changed_) > 4:
                        print(sorted(changed_))
                        print([k_old, k_new, K_old, self.clusters.K])
                        breakpoint()

                    for k in changed_:
                        log_post_Z[k] =  self.clusters.get_posterior_probability_Z_k(k)
                    
                    posteriorList.append(np.sum(log_post_Z))

                    # Calculate the ARI if true assignments are provided
                    if len(self.trueZ) != 0:
                        ARI_list.append(round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3))    

            ####### Feature Selection ###########

            # Check if feature selection (FS) is enabled
            if self.FS:
                # Set the lambda parameter
                lamb = 200
                
                # Calculate log probabilities for unimportant and important features
                log_prob_unimp = self.clusters.log_prob_unimp_marginal(lamb)
                log_prob_imp = self.clusters.log_prob_imp_marginal(lamb)

                # Iterate over each cluster
                for k in range(self.clusters.K):
                    # Iterate over each feature
                    for j in range(self.clusters.D):
                        # Compare log probabilities and select features accordingly
                        if log_prob_unimp[k][j] == log_prob_imp[k][j]:
                            self.clusters.features[k][j] = 1
                        else:
                            self.clusters.features[k][j] = np.argmax(np.array([log_prob_unimp[k][j], log_prob_imp[k][j]]) + np.random.gumbel(0, 1, 2))

            # Update cluster assignments
            new_assignments = self.clusters.assignments

            # Check which assignments have changed
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(self.clusters.N):
                if not assignments_change[i]:
                    changed_clusters.append(old_assignments[i])
                    changed_clusters.append(new_assignments[i])
            changed_clusters = list(set(changed_clusters))

            # Update posterior probabilities for changed clusters
            for k in changed_clusters:
                log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

            # Sum the log posterior probabilities
            sum_log_post_Z = np.sum(log_post_Z)

            # Update the maximum log posterior if the current one is greater
            if sum_log_post_Z > self.log_max_post:
                self.log_max_post = sum_log_post_Z
                self.z_map = self.clusters.assignments.copy()
                self.iter_map = i_iter + 1

            # Check if the posterior probability has changed
            if sum_log_post_Z != ass_posterior:
                same_posterior_count = 0
                ass_posterior = sum_log_post_Z
            else:
                same_posterior_count += 1

            # Print results if the toPrint flag is set
            if toPrint:
                if len(self.trueZ) != 0:
                    print(f"\nrun: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{round(sum_log_post_Z, 3)}, ARI: {adjusted_rand_score(self.trueZ, self.clusters.assignments.copy())}")
                    if self.FS:
                        print("features:\n", self.clusters.features)
                else:
                    print(f"\nrun: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{round(sum_log_post_Z, 3)}")
                    # Uncomment the following line if you want to print features
                    # print("features:\n", self.clusters.features)

            # Uncomment the following line if you need a breakpoint for debugging
            # breakpoint()

            # Stop the iterations if the posterior hasn't changed for several iterations
            if same_posterior_count > 3:
                break

            # Print iteration progress
            print(f"{i_iter}/{n_iter}               ", end='\r')

        # Calculate Bayesian Information Criterion (BIC)
        self.BIC = self.clusters.get_no_free_param() - 2 * self.log_max_post

        # Print the final results for the run
        print(f"\nRun: {run_id + 1}, K:{len(set(self.z_map))}, BIC: {self.BIC}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")

        # Print final feature selection results
        print("Final features:")
        print(self.clusters.features)

        # Prepare data for post-processing
        postData = {
            "run": run_id,
            "n_iter": n_iter,
            "posterior": posteriorList,
            "ARI": ARI_list
        }

        # Return post-processing data
        return postData




if __name__ == "__main__":
    
    # Start timer to measure model runtime
    model_start_time = time.perf_counter()
    
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing Gaussian mixture data")
    parser.add_argument("-k", required=False, type=int, help="Known K or maximum number of clusters if K is unknown")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs with different initial assignments")
    parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the true parameters file (non-pickle file)")
    parser.add_argument("-p", required=False, action="store_true", help="Print results during Gibbs sampling")
    parser.add_argument("-seed", required=False, type=int, help="Set a seed value")
    parser.add_argument("-fs", required=False, action="store_true", help="Enable feature selection")
    parser.add_argument("-fd", required=False, type=argparse.FileType('r'), help="Path to the file containing true feature importance")

    # Parse command-line arguments
    args = parser.parse_args()

    # Set global random seed
    global_seed = np.random.randint(1, 2**31 - 1) if args.seed is None else args.seed
    np.random.seed(global_seed)

    ################################## Extract data ##################################

    # Read data from the input file
    X = []
    dataFile = args.f
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        X.append(np.array([float(i) for i in line.strip().split(',')]))
    X = np.array(X)

    # Uncomment the following line if you need to exclude some initial columns
    # X = X[:, 4:]
    
    N = len(X)  # Number of data points
    D = len(X[0])  # Number of dimensions/features
    
    # Set model parameters
    K_max_BIC = 50 if args.k is None else args.k
    n_iter = 50 if args.i is None else args.i
    training_runs = 1 if args.r is None else args.r

    # Print initial setup information
    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, D: {D}, K: {K_max_BIC}, Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set hyper-parameters ##################################
    # Set hyperparameters for the Gaussian mixture model
    alpha = 1.0 
    m_0 = np.zeros(D)
    k_0 = 0.1 
    v_0 = 1
    S_0 = v_0 * np.ones(D)
    
    # Initialize prior object
    prior = utils.NIchi2(m_0, k_0, v_0, S_0)

    ################################## Model ##################################
    
    # Check if results should be printed during Gibbs sampling
    toDisplay = args.p
    
    # Check if feature selection is enabled
    FS = args.fs

    # Load true feature importance if provided
    true_features = []
    if args.fd:
        featureFile = args.fd
        for line in featureFile:
            true_features.append(np.array([int(i) for i in line.strip().split(',')]))
        true_features = np.array(true_features)

    # Print total number of training runs
    print(f"Total training runs: {training_runs}")

    # Load true assignments if provided
    trueFile = args.t
    if trueFile:
        trueAssignments = np.array([int(line.strip()) for line in trueFile])
    else:
        trueAssignments = []

    # Initialize variables to track the best model
    max_post = -1 * np.inf
    least_BIC = -1 * np.inf

    # Perform multiple training runs
    for i in range(training_runs):
        print(f"\nRun:  {i + 1}")

        # Generate random starting assignments
        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)

        # Uncomment and modify the following lines if you want to use specific starting assignments
        # starting_assignments = trueAssignments
        # params_true = pickle.load(open("../data_n1000_d10_k10_m2.0_c2.1_catD0_catM4_seed1616.trueParamPickle", "rb"))
        # starting_assignments = params_true['z']
        # starting_assignments = params_true = np.array(json.load(open("../Z_true.json", "rb"))['z'])
        # starting_assignments = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))['z']
        
        # Initialize Bayesian GMM with feature selection
        bayesgmm = bayesGMM_FS(X, prior, alpha, starting_assignments, FS, features=true_features)
        
        # Run Gibbs sampling
        bayesgmm.gibbs_sampler(n_iter, i, trueAssignments=trueAssignments, toPrint=toDisplay, greedyRun=False, savePosterior=False)
        
        # Track the best model based on BIC score
        if bayesgmm.BIC > least_BIC:
            least_BIC = bayesgmm.BIC
            best_bayesgmm = bayesgmm

    ################################## Model results ##################################

    # Get predictions from the best model
    z_pred_map = best_bayesgmm.z_map
    predicted_K = len(set(z_pred_map))

    # Print results of the best model
    print(f"\nBest Model:\nlogmax posterior: {best_bayesgmm.log_max_post}\nBIC: {best_bayesgmm.BIC}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_bayesgmm.run_id + 1} iteration: {best_bayesgmm.iter_map}")
    print(f"Time: {time.perf_counter() - model_start_time}")
    
    mu_pred = []
    sigma_pred = []

    # Prepare predictions for saving
    preds = {
        "mu": mu_pred,
        "sigma": np.array(sigma_pred),
        "z": z_pred_map,
        "time": time.perf_counter() - model_start_time,
        "z_last_iter": best_bayesgmm.clusters.assignments
    }
    
    ################################## Save results ##################################

    # Determine output directory
    if args.o is None:
        if args.fs:
            outDir = "outGaussFS"
        else:
            outDir = "outGaussNoisey"
    else:
        outDir = args.o

    # Prepare output file paths
    outputFileName = f"{dataFilename}"    
    outputFilePath = f"{outDir}/{outputFileName}.txt"

    # Save results in a readable format
    with open(outputFilePath, "w") as wFile:
        wFile.write(f"N: {N}\n")
        wFile.write(f"D: {D}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {global_seed}\n")
        wFile.write(f"Iterations: {n_iter}\n")
        wFile.write(f"alpha: {alpha}\n")
        wFile.write(f"time: {time.perf_counter() - model_start_time}\n")
        wFile.write(f"BIC score: {best_bayesgmm.BIC}\n")
        wFile.write(f"log max posterior: {best_bayesgmm.log_max_post}\n")
        wFile.write(f"MAP assignments: {best_bayesgmm.z_map}\n")
        wFile.write(f"Last iteration assignments: {best_bayesgmm.clusters.assignments}\n")
        wFile.write("m_0:")
        np.savetxt(wFile, m_0)
        wFile.write(f"k_0: {k_0}\n")
        wFile.write(f"v_0: {v_0}\n")
        wFile.write("S_0:")
        np.savetxt(wFile, S_0)

    # Save predictions in a pickle file
    outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)
    
    # Save cluster assignments in a separate file
    with open(f"{outDir}/{outputFileName}.labels", "w") as ff:
        for z_i in z_pred_map:
            ff.write(f"{z_i}\n")

    # Save feature importance in a separate file
    with open(f'{outDir}/{outputFileName}.features', 'w') as ff1:
        for data_vec in best_bayesgmm.clusters.features:
            data_vec = [str(i) for i in data_vec]
            ff1.write(",".join(data_vec) + '\n')

    # Print locations of the saved results
   
    print(f"The encoded results are saved in: {outDir}/{outputFileName}.p\n")
    print(f"The readable feature importance are saved in: {outDir}/{outputFileName}.features\n")
    print(f"The readable results are saved in: {outputFilePath}\n")


