from tracemalloc import start
import numpy as np
from cluster_stats_new import categoricalClusters
import utils
import time
import argparse
import pickle
import os
import time
from sklearn.metrics.cluster import adjusted_rand_score


class catMM():
    """
    Categorical Mixture Model (GMM) with each feature being independent.

    This class implements a Gibbs sampler for Bayesian CatMM.
    It initializes the model with given data, prior, and initial cluster assignments, and provides
    a method to run the Gibbs sampler for a specified number of iterations.
    """

    def __init__(self, C: int, alpha: float, gamma, assignments: int, isTrueZ=0):

        """
        Initialize the Categorical MM.

        Args:
            C (np.array, dtype=float): 2D NumPy array of shape (n_samples, n_features) containing the categorical data.
            alpha (float): Dirichlet hyperparameter for mixing probabilities, alpha_0.
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
        self.N, catD = C.shape

        # Get number of categories for each feature
        self.Ms = np.zeros(catD, int)
        for d in range(catD):
            self.Ms[d] = len(set(C[d]))

        # Setting up the Categorical Cluster object which will track the features and component-wise statistics
        self.clusters = categoricalClusters(C, alpha, gamma, K, assignments)
        
        # Initializing the outputs
        self.z_map = assignments
        self.iter_map = 0
        self.log_max_post = -1*np.inf
        self.BIC = 0.
        self.run_id = -1

    def gibbs_sampler(self, n_iter: int, run_id: int,  toPrint=True, savePosterior=False, trueAssignments=[]):

        """
        Run the Gibbs sampler for the Bayesian GMM.

        Args:
            n_iter (int): Number of iterations to run the Gibbs sampler.
            run_id (int): Unique identifier for the current run.
            toPrint (bool, optional): If True, print the results for each iteration. Default is True.
            savePosterior (bool, optional): If True, save the posterior score for each data step in each iteration. Default is False.
            trueAssignments (list, optional): Ground truth cluster assignments for calculating Adjusted Rand Index (ARI). Default is an empty list.
        
        """

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
                
                # Cache the previous cluster statistics if the same cluster is assigned to the current data point
                k_old = self.clusters.assignments[i]
                K_old = self.clusters.K
                stats_old = self.clusters.cache_cluster_stats(k_old)

                # Remove the data point from the data
                self.clusters.del_assignment(i)
                
                # Calculate f(z_i = k | z_[-i], alpha)
                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(self.N + self.alpha - 1)

                # Calculate f(c_i | C[-i], z_i = k, z_[-i], Gamma)
                log_prob_c_i = self.clusters.log_post_pred(i)
                
                # Get f(z_i = k | z_[-i])
                log_prob_z_k = log_prob_z_k_alpha + log_prob_c_i

                # Sample new cluster identity for the data point using Gumbel-max trick
                k = utils.sample_numpy_gumbel(log_prob_z_k)
                # k = utils.sample(log_prob_z_k)
                
                # if an empty cluster is sampled
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
                    new_assignments = self.clusters.assignments
                    assignments_change = old_assignments == new_assignments
                    changed_clusters = []
                    for i in range(self.N):
                        if not assignments_change[i]:
                            changed_clusters.append(old_assignments[i])
                            changed_clusters.append(new_assignments[i])
                    changed_clusters = list(set(changed_clusters))

                    for j in changed_clusters:
                        log_post_Z[j] = self.clusters.get_posterior_probability_Z_k(j)

                    posteriorList.append(np.sum(log_post_Z))
                    
                    # Calculate the ARI if true assignments are provided
                    if len(self.trueZ) != 0:
                        ARI_list.append(round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3))    

            # Get the list of all changed clusters for the iteration
            new_assignments = self.clusters.assignments
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(self.N):
                if not assignments_change[i]:
                    changed_clusters.append(old_assignments[i])
                    changed_clusters.append(new_assignments[i])
            changed_clusters = list(set(changed_clusters))

            # Get the posterior score
            for j in changed_clusters:
                log_post_Z[j] = self.clusters.get_posterior_probability_Z_k(j)
            sum_log_post_Z = np.sum(log_post_Z)

            # Change the MAP parameters to be updated
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
            
            
            if same_posterior_count > n_iter:
                break

            print(f"{i_iter}/{n_iter}               ",end='\r')

        self.BIC = self.clusters.K*(self.Ms.sum()) * np.log(self.N) - (2 * self.log_max_post)

        print(f"\nRun: {run_id + 1}, K:{len(set(self.z_map))}, BIC: {self.BIC}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")
        
        postData = {
            "run":run_id,
            "n_iter":n_iter,
            "posterior":posteriorList,
            "ARI":ARI_list
        }

        return postData
    
if __name__ == "__main__":
    model_start_time = time.perf_counter()

    # Setup argument parser
    parser = argparse.ArgumentParser()

    # Define the required and optional arguments for the script
    parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing gauusian mixture data")
    parser.add_argument("-k", required=True, type=int, help="Known K and if it's unknown Maximum number of clusters (Or your guess that the number of clusters can't be more than that)")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs to run with different initial assignments")
    parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the true parameters file (non-pickle file)")
    parser.add_argument("-p", required=False, action="store_true", help="Will print results while Gibbs sampling")
    parser.add_argument("-seed", required=False, type=int, help="set a seed value")

    # Parse arguments
    args = parser.parse_args()

    # Set random seed
    global_seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed
    np.random.seed(global_seed)

    ##################################  Extract data ##################################
    
    # Read data from the input file
    C = []
    dataFile = args.f
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        C.append(np.array([int(float(i)) for i in line.strip().split(',')]))
    C = np.array(C)

    N, catD = C.shape

    Ms = np.zeros(catD, int)
    for d in range(catD):
        Ms[d] = len(set(C[d]))
    
    # Model parameters
    K_max_BIC = args.k
    n_iter = 50 if args.i == None else args.i

    training_runs = 1 if args.r == None else args.r

    # Print initial setup information
    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, K: {K_max_BIC}, Ms: {Ms} Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set hyper-parameters  ################################## (can we look at the data to set hyperparameters?)
    # Set hyperparameters for the model
    alpha = 1.0
    gamma = 0.2

    ################################## Model ##################################
    print(f"Total training runs: {training_runs}")

    trueFile = args.t
    if trueFile:
        trueAssignments = np.array([int(line.strip()) for line in trueFile])
        bayesgmm = catMM(C, alpha, gamma, trueAssignments, 1)
        bayesgmm.gibbs_sampler(n_iter, -1)
    else:
        trueAssignments = []

    # Initialize variables to track the best model
    max_post = -1*np.inf
    least_BIC = 1*np.inf

    # Run training with different initial assignments
    for i in range(training_runs):
        print(f"\nRun:  {i+1}")

        # Ensure unique initial assignments
        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)

        # Uncomment and modify the following lines if you want to use specific starting assignments
        # params_true = pickle.load(open("../data_n1000_d10_k10_m2.0_c2.1_catD0_catM4_seed1616.trueParamPickle", "rb"))
        # starting_assignments = params_true['z']
        # starting_assignments = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))['z']
        # starting_assignments = np.array([3, 0, 2, 0, 0, 3, 2, 2, 3, 3, 0, 2, 2, 3, 0, 0, 0, 2, 2, 2, 3, 0, 3, 0, 2, 0, 0, 3, 3, 0, 2, 1, 2, 2, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 2, 2, 3, 3, 0, 3, 2, 2, 3, 0, 3, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 2, 0, 3, 2, 2, 0, 0, 2, 3, 2, 0, 3, 2, 0, 0, 3, 3, 0, 3, 0, 0, 0, 1, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 3, 0, 2, 3, 0, 2, 3, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 0, 1, 3, 2, 2, 0, 3, 0, 0, 2, 0, 3, 2, 0, 2, 2, 3, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 3, 2, 0, 2, 0, 0, 3, 0, 2, 0, 2, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 2, 3, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 1, 0, 3, 0, 0, 0, 2, 0, 2, 0, 3, 3, 0, 0, 0, 3, 0, 0, 3, 3, 3, 0, 3, 0, 3, 2, 2, 2, 0, 3, 0, 0, 3, 0, 2, 0, 0, 3, 0, 0, 3, 3, 1, 2, 1, 0, 2, 0, 2, 3, 2, 3, 0, 3, 0, 3, 2, 0, 0, 0, 0, 0, 2, 3, 3, 3, 0, 3, 0, 0, 0, 3, 3, 3, 0, 2, 0, 0, 3, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 3, 0, 2, 2, 3, 3, 2, 0, 0, 3, 2, 2, 2, 2, 3, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 0, 2, 0, 3, 0, 2, 2, 2, 0, 0, 3, 0, 2, 3, 3, 2, 2, 2, 0, 2, 2, 0, 0, 3, 2, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 3, 2, 3, 2, 0, 2, 0, 0, 0, 0, 3, 3, 0, 3, 2, 0, 0, 0, 3, 0, 2, 3, 3, 0, 0, 2, 0, 0, 0, 0, 3, 0, 2, 3, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 3, 2, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 3, 2, 0, 2, 0, 0, 3, 3, 0, 2, 0, 2, 2, 0, 3, 0, 0, 3, 2, 2, 0, 2, 3, 0, 3, 2, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 3, 2, 0, 2, 0, 0, 3, 0, 3, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 0, 0, 2, 0, 2, 2, 0, 3, 0, 0, 3, 3, 2, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 2, 2, 0, 0, 2, 0, 1, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 3, 0, 0, 0, 2, 3, 2, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 2, 3, 0, 3, 3, 3, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 3, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 2, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 2, 3, 0, 0, 0, 0, 3, 2, 0, 2, 2, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3, 2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 3, 2, 0, 0, 0, 2, 0, 3, 2, 0, 0, 3, 0, 0, 2, 2, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 0, 3, 0, 3, 3, 0, 2, 3, 2, 3, 0, 2, 3, 3, 0, 3, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 3, 3, 0, 2, 0, 2, 2, 0, 2, 0, 0, 3, 2, 3, 0, 0, 3, 3, 2, 3, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 2, 3, 0, 3, 3, 0, 2, 3, 3, 2, 0, 2, 2, 0, 0, 0, 1, 2, 2, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2, 3, 0, 3, 0, 3, 3, 2, 0, 0, 0, 3, 0, 2, 2, 0, 2, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 3, 2, 3, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 2, 0, 2, 2, 2, 0, 2, 0, 2, 2, 3, 0, 2, 2, 2, 3, 0, 0, 2, 2, 2, 0, 0, 2, 2, 0, 3, 0, 2, 2, 2, 3, 0, 2, 3, 0, 0, 2, 3, 0, 3, 2, 0, 0, 0, 2, 3, 0, 3, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 3, 3, 0, 2, 3, 0, 2, 2, 2, 0, 3, 2, 0, 0, 3, 3, 2, 0, 3, 2, 0, 1, 0, 3, 3, 2, 0, 3, 0, 0, 3, 3, 0, 2, 2, 3, 0, 2, 2, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 3, 0, 2, 2, 3, 0, 2, 3, 2, 2, 0, 0, 2, 0, 0, 0, 3, 2, 2, 2, 2, 2, 3, 2, 0, 0, 3, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 2, 0, 0, 0])
        # starting_assignments = pickle.load(open("catData4d2.p", "rb"))['z']

        # Initialize and run the CatMM
        catmm = catMM(C, alpha, gamma, starting_assignments)
        catmm.gibbs_sampler(n_iter, i, trueAssignments=trueAssignments)

        # Track the best model based on BIC score
        if catmm.BIC < least_BIC:
            least_BIC = catmm.BIC
            best_catmm =catmm


    ##################################  Model results ##################################

    # Get predictions from the best model
    z_pred_map = best_catmm.z_map
    predicted_K = len(set(z_pred_map))

    # Print results of the best model
    print(f"\nBest Model:\nlogmax posterior: {best_catmm.log_max_post}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_catmm.run_id + 1} iteration: {best_catmm.iter_map}")
    print(f"Time: {time.perf_counter() - model_start_time}")

    # Store predictions
    preds = {
        "z": z_pred_map,
        "time": time.perf_counter() - model_start_time,
        "z_last_iter": best_catmm.clusters.assignments
    }
    
    ##################################  Save results ##################################

    outDir = "outCat" if args.o == None else args.o

    if outDir not in os.listdir():
        os.mkdir(outDir)
    
    outputFileName = f"{dataFilename}"    
    outputFilePath = f"{outDir}/{outputFileName}.txt"

    # Save results to text file
    with open(outputFilePath, "w") as wFile:
        wFile.write(f"N: {N}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {global_seed}\n")
        wFile.write(f"Iterations: {n_iter}\n")
        wFile.write(f"alpha: {alpha}\n")
        wFile.write(f"time: {time.perf_counter() - model_start_time}\n")

        wFile.write(f"BIC score: {best_catmm.BIC}\n")
        wFile.write(f"log max posterior: {best_catmm.log_max_post}\n")
        wFile.write(f"MAP assignments: {best_catmm.z_map}\n")
        wFile.write(f"Last iteration assignments: {best_catmm.clusters.assignments}\n")
    
        wFile.write(f"gamma:{gamma}")

    # Save predictions to pickle file
    outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

    # Print locations of the saved results
    print(f"The encoded results are saved in: {outDir}/{outputFileName}.p\n")
    print(f"The readable results are saved in: {outputFilePath}\n")




################################################################################################################
################## TRASH CODE ############################################ TRASH CODE ##########################
################################################################################################################
    
# from tracemalloc import start
# import numpy as np
# from cluster_stats_new import categoricalClusters
# import utils
# import time
# import argparse
# import pickle
# import os
# import time
# from sklearn.metrics.cluster import adjusted_rand_score
# import json

# class catMM():
#     def __init__(self, C, alpha, gamma, seed, assignments):
#         self.alpha = alpha
#         K = len(set(assignments))
#         self.K_max = K
#         self.seed = seed
        
#         self.clusters = categoricalClusters(C, alpha, gamma, K, assignments)
 
#         self.z_map = assignments
#         self.iter_map = 0
#         self.log_max_post = -1*np.inf
#         self.BIC = 0.
#         self.run_id = -1

#     def gibbs_sampler(self, n_iter, run_id):
#         self.run_id = run_id

#         same_posterior_count = 0
#         ass_posterior = 0

#         log_post_Z = np.zeros(self.K_max)
#         for k in range(self.K_max):
#             log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

#         # params_true = json.load(open("../Z_true.json", "rb"))
#         # params_true = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))
#         params_true = pickle.load(open("catData4.p", "rb"))
#         print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{np.sum(log_post_Z)}, ARI: {round(adjusted_rand_score(params_true['z'], self.clusters.assignments), 3)}")

#         for i_iter in range(n_iter):
#             old_assignments = self.clusters.assignments.copy()

#             for i in range(self.clusters.N):
                
#                 k_old = self.clusters.assignments[i]
#                 K_old = self.clusters.K
#                 stats_old = self.clusters.cache_cluster_stats(k_old)

#                 self.clusters.del_assignment(i)
                
#                 log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(N + self.alpha - 1)

#                 log_prob_c_i = self.clusters.log_post_pred(i)
                
#                 log_prob_z_k = log_prob_z_k_alpha + log_prob_c_i

#                 # k = utils.sample(log_prob_z_k)
#                 k = utils.sample_numpy_gumbel(log_prob_z_k)
                
#                 # if an empty cluster is sampled
#                 if k >= self.clusters.K:
#                     k = self.clusters.K
            
#                 # breakpoint()
#                 # if the same old assignment is sampled AND deleting i-th data point didn't make the cluser empty
#                 if k==k_old and self.clusters.K == K_old:
#                     self.clusters.restore_cluster_stats(k_old, *stats_old)
#                     self.clusters.assignments[i] = k_old

#                 else:
#                     self.clusters.add_assignment(i,k)


#             new_assignments = self.clusters.assignments
#             assignments_change = old_assignments == new_assignments
#             changed_clusters = []
#             for i in range(N):
#                 if not assignments_change[i]:
#                     changed_clusters.append(old_assignments[i])
#                     changed_clusters.append(new_assignments[i])
#             changed_clusters = list(set(changed_clusters))

#             for j in changed_clusters:
#                 log_post_Z[j] = self.clusters.get_posterior_probability_Z_k(j)

#             sum_log_post_Z = np.sum(log_post_Z)

#             if sum_log_post_Z > self.log_max_post:
#                 self.log_max_post = sum_log_post_Z
#                 self.z_map = self.clusters.assignments.copy()
#                 self.iter_map = i_iter + 1

#             if sum_log_post_Z != ass_posterior:
#                 same_posterior_count = 0
#                 ass_posterior = sum_log_post_Z
#             else:
#                 same_posterior_count += 1

#             # params_true = json.load(open("../Z_true.json", "rb"))
#             if (i_iter + 1) % 10 == 0:
#                 params_true = pickle.load(open("catData4.p", "rb"))
#                 print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{sum_log_post_Z}, ARI: {round(adjusted_rand_score(params_true['z'], self.clusters.assignments), 3)}")
#                 # print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{sum_log_post_Z}, ARI: {adjusted_rand_score(params_true['z'], self.clusters.assignments)}, ARI max post: {round(adjusted_rand_score(params_true['z'], self.z_map), 2)}")
            
#             if same_posterior_count > n_iter:
#                 break
#             print(f"{i_iter}/{n_iter}               ",end='\r')

#         # self.BIC = self.clusters.K*(M) * np.log(N) - (2 * self.log_max_post)

#         print(f"\nRun: {run_id + 1}, Seed: {self.seed}, K:{len(set(self.z_map))}, BIC: {self.BIC}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")
        
# if __name__ == "__main__":
#     model_start_time = time.perf_counter()

#     parser = argparse.ArgumentParser()

#     parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing gauusian mixture data")
#     parser.add_argument("-k", required=True, type=int, help="Known K and if it's unknown Maximum number of clusters (Or your guess that the number of clusters can't be more than that)")
#     parser.add_argument("-o", required=False, type=str, help="Output directory")
#     parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
#     parser.add_argument("-r", required=False, type=int, help="Number of training runs to run with different initial assignments")
#     parser.add_argument("-seed", required=False, type=int, help="set a seed value")

#     args = parser.parse_args()

#     global_seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed
#     # seed = 82   
#     np.random.seed(global_seed) # should not be same as in learn file
#     # np.random.seed(np.random.randint(1, 2**31 - 1))

#     ##################################  Extract data ##################################
#     C = []
#     dataFile = args.f
#     dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
#     for line in dataFile:
#         C.append(np.array([float(i) for i in line.strip().split(',')]))

#     N = len(C[0])
#     catD = len(C)
#     Ms = np.zeros(catD, int)
#     for d in catD:
#         M[d] = len(set(C[d]))

#     # model parameters
#     K_max_BIC = args.k
#     n_iter = 50 if args.i == None else args.i
#     training_runs = 1 if args.r == None else args.r

#     print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
#     print(f"N: {N}, K: {K_max_BIC}, M: {Ms.tolist()} Iterations: {n_iter}, Global seed: {global_seed}\n")

#     ################################## Set hyper-parameters  ################################## (can we look at the data to set hyperparameters?)
#     alpha = 1.0
#     gamma = 0.2

#     ################################## Model ##################################
#     seed_l = np.random.randint(1, 2**31 -1, training_runs)
#     # seed_l = np.arange(1, training_runs + 1)
#     print(f"Total training runs: {training_runs}")

#     max_post = -1*np.inf
#     least_BIC = 1*np.inf
#     for i in range(training_runs):
#         seed = seed_l[i]
#         print(f"\nRun:  {i+1}, seed: {seed}")
#         np.random.seed(seed)

#         starting_assignments = []
#         while len(set(starting_assignments)) != K_max_BIC:
#             starting_assignments = np.random.randint(0, K_max_BIC, N)

#         # params_true = pickle.load(open("../data_n1000_d10_k10_m2.0_c2.1_catD0_catM4_seed1616.trueParamPickle", "rb"))
#         # starting_assignments = params_true['z']
#         # starting_assignments = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))['z']
#         # starting_assignments = np.array([3, 0, 2, 0, 0, 3, 2, 2, 3, 3, 0, 2, 2, 3, 0, 0, 0, 2, 2, 2, 3, 0, 3, 0, 2, 0, 0, 3, 3, 0, 2, 1, 2, 2, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 2, 2, 3, 3, 0, 3, 2, 2, 3, 0, 3, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 2, 0, 3, 2, 2, 0, 0, 2, 3, 2, 0, 3, 2, 0, 0, 3, 3, 0, 3, 0, 0, 0, 1, 0, 2, 3, 0, 0, 3, 0, 0, 3, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 3, 0, 2, 3, 0, 2, 3, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 0, 1, 3, 2, 2, 0, 3, 0, 0, 2, 0, 3, 2, 0, 2, 2, 3, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 3, 2, 0, 2, 0, 0, 3, 0, 2, 0, 2, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 2, 3, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 1, 0, 3, 0, 0, 0, 2, 0, 2, 0, 3, 3, 0, 0, 0, 3, 0, 0, 3, 3, 3, 0, 3, 0, 3, 2, 2, 2, 0, 3, 0, 0, 3, 0, 2, 0, 0, 3, 0, 0, 3, 3, 1, 2, 1, 0, 2, 0, 2, 3, 2, 3, 0, 3, 0, 3, 2, 0, 0, 0, 0, 0, 2, 3, 3, 3, 0, 3, 0, 0, 0, 3, 3, 3, 0, 2, 0, 0, 3, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 3, 0, 2, 2, 3, 3, 2, 0, 0, 3, 2, 2, 2, 2, 3, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 3, 0, 2, 0, 3, 0, 2, 2, 2, 0, 0, 3, 0, 2, 3, 3, 2, 2, 2, 0, 2, 2, 0, 0, 3, 2, 0, 0, 2, 0, 2, 3, 2, 0, 0, 0, 0, 3, 2, 3, 2, 0, 2, 0, 0, 0, 0, 3, 3, 0, 3, 2, 0, 0, 0, 3, 0, 2, 3, 3, 0, 0, 2, 0, 0, 0, 0, 3, 0, 2, 3, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 3, 2, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 3, 2, 0, 2, 0, 0, 3, 3, 0, 2, 0, 2, 2, 0, 3, 0, 0, 3, 2, 2, 0, 2, 3, 0, 3, 2, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 3, 2, 0, 2, 0, 0, 3, 0, 3, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 0, 0, 2, 0, 2, 2, 0, 3, 0, 0, 3, 3, 2, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 2, 2, 0, 0, 2, 0, 1, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 3, 0, 0, 0, 2, 3, 2, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 2, 3, 0, 3, 3, 3, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 3, 0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 2, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 2, 3, 0, 0, 0, 0, 3, 2, 0, 2, 2, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3, 2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 3, 2, 0, 0, 0, 2, 0, 3, 2, 0, 0, 3, 0, 0, 2, 2, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 0, 3, 0, 3, 3, 0, 2, 3, 2, 3, 0, 2, 3, 3, 0, 3, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 0, 3, 3, 0, 2, 0, 2, 2, 0, 2, 0, 0, 3, 2, 3, 0, 0, 3, 3, 2, 3, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 2, 3, 0, 3, 3, 0, 2, 3, 3, 2, 0, 2, 2, 0, 0, 0, 1, 2, 2, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2, 3, 0, 3, 0, 3, 3, 2, 0, 0, 0, 3, 0, 2, 2, 0, 2, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 3, 2, 3, 0, 2, 0, 2, 0, 0, 3, 0, 3, 0, 2, 0, 2, 2, 2, 0, 2, 0, 2, 2, 3, 0, 2, 2, 2, 3, 0, 0, 2, 2, 2, 0, 0, 2, 2, 0, 3, 0, 2, 2, 2, 3, 0, 2, 3, 0, 0, 2, 3, 0, 3, 2, 0, 0, 0, 2, 3, 0, 3, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 3, 3, 0, 2, 3, 0, 2, 2, 2, 0, 3, 2, 0, 0, 3, 3, 2, 0, 3, 2, 0, 1, 0, 3, 3, 2, 0, 3, 0, 0, 3, 3, 0, 2, 2, 3, 0, 2, 2, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 3, 0, 2, 2, 3, 0, 2, 3, 2, 2, 0, 0, 2, 0, 0, 0, 3, 2, 2, 2, 2, 2, 3, 2, 0, 0, 3, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 2, 0, 0, 0])
        
#         # starting_assignments = pickle.load(open("catData4.p", "rb"))['z']

#         catmm = catMM(C, alpha, gamma, seed, starting_assignments)
#         catmm.gibbs_sampler(n_iter, i)

#         # if catmm.BIC < least_BIC:
#         #     least_BIC = catmm.BIC
#         #     best_catmm =catmm

#         if catmm.log_max_post > max_post:
#             max_post = catmm.log_max_post
#             best_catmm = catmm

#     ##################################  Model results ##################################

#     z_pred_map = best_catmm.z_map
#     predicted_K = len(set(z_pred_map))

#     print(f"\nBest Model:\nlogmax posterior: {best_catmm.log_max_post}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_catmm.run_id + 1} iteration: {best_catmm.iter_map}")
#     print(f"Time: {time.perf_counter() - model_start_time}")

#     mu_pred = []
#     sigma_pred = []

#     preds = {
#         "mu": mu_pred,
#         "sigma": np.array(sigma_pred),
#         "z": z_pred_map,
#         "time": time.perf_counter() - model_start_time,
#         "z_last_iter": best_catmm.clusters.assignments
#     }
    
#     ##################################  Save results ##################################

#     outDir = "outputs_result" if args.o == None else args.o

#     if outDir not in os.listdir():
#         os.mkdir(outDir)
    
#     outputFileName = f"{dataFilename}"    
#     outputFilePath = f"{outDir}/{outputFileName}.txt"

#     with open(outputFilePath, "w") as wFile:
#         wFile.write(f"N: {N}\n")
#         wFile.write(f"Ms: {Ms}\n")
#         wFile.write(f"K: {predicted_K}\n\n")
#         wFile.write(f"Seed: {catmm.seed}\n")
#         wFile.write(f"Iterations: {n_iter}\n")
#         wFile.write(f"alpha: {alpha}\n")
#         wFile.write(f"time: {time.perf_counter() - model_start_time}\n")

#         wFile.write(f"BIC score: {best_catmm.BIC}\n")
#         wFile.write(f"log max posterior: {best_catmm.log_max_post}\n")
#         wFile.write(f"MAP assignments: {best_catmm.z_map}\n")
#         wFile.write(f"Last iteration assignments: {best_catmm.clusters.assignments}\n")

#         wFile.write(f"gamma:{gamma}")

#     outputFile = open(f"{outDir}/{outputFileName}.predParamPickle", "wb")
#     pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

#     print(f"The encoded results are saved in: {outDir}/{outputFileName}.predParamPickle\n")
#     print(f"The readable results are saved in: {outputFilePath}\n")