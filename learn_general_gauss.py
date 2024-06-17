import numpy as np
from cluster_stats_new import gaussianClusters
import utils
import time
import argparse
import pickle
import os
import time
from sklearn.metrics.cluster import adjusted_rand_score

# Define a class for Bayesian Gaussian Mixture Model (GMM)
class bayesGMM():
    def __init__(self, X: float, prior, alpha:float, seed:int, assignments:int):
        # Initialize the class with the dataset X, prior, alpha parameter, random seed, and initial assignments
        self.alpha = alpha
        N, D = X.shape  # Number of data points and dimensionality of the data

        K = len(set(assignments))  # Number of initial clusters
        self.K_max = K  # Maximum number of clusters

        self.seed = seed 

        # Initialize clusters with the given data, prior, alpha, number of clusters, and initial assignments
        self.clusters = gaussianClusters(X, prior, alpha, K, assignments)
        self.z_map = assignments  # Store the current cluster assignments
        self.iter_map = 0  # Iteration at which the maximum posterior was found
        self.log_max_post = -1 * np.inf  # Initialize log of maximum posterior
        self.BIC = 0.  # Initialize BIC score
        self.run_id = -1  # ID of the current run

    # Gibbs sampler for Bayesian inference
    def gibbs_sampler(self, n_iter, run_id):
        self.run_id = run_id
        
        np.random.seed(self.seed)

        same_posterior_count = 0  # Count iterations with the same posterior
        ass_posterior = 0  

        log_post_Z = np.zeros(self.K_max)  # Log posterior probabilities of clusters
        for k in range(self.K_max):
            log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

        # Print initial state
        print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{np.sum(log_post_Z)}", end="\r")

        # Loop through each itration
        for i_iter in range(n_iter):
            old_assignments = self.clusters.assignments.copy()  # Copy current assignments

            # Loop through each data point
            for i in range(self.clusters.N):
                k_old = self.clusters.assignments[i]  # Old cluster assignment for data point i
                K_old = self.clusters.K  # Old number of clusters
                stats_old = self.clusters.cache_cluster_stats(k_old)  # Cache stats for old cluster

                self.clusters.del_assignment(i)  # Remove data point i from its cluster
                
                # Compute log probabilities for each cluster
                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max) - np.log(N + self.alpha - 1)
                log_prob_x_i = self.clusters.log_post_pred(i)
                log_prob_z_k = log_prob_z_k_alpha + log_prob_x_i

                k = utils.sample_numpy_gumbel(log_prob_z_k)  # Sample new cluster assignment using Gumbel-max trick

                if k >= self.clusters.K:
                    k = self.clusters.K  # If sampled cluster is new, add it

                if k == k_old and self.clusters.K == K_old:
                    # Restore old cluster stats if assignment hasn't changed
                    self.clusters.restore_cluster_stats(k_old, *stats_old)
                    self.clusters.assignments[i] = k_old
                else:
                    self.clusters.add_assignment(i, k)  # Add data point i to the new cluster

            # Determine changes in cluster assignments
            new_assignments = self.clusters.assignments
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(N):
                if not assignments_change[i]:
                    changed_clusters.append(old_assignments[i])
                    changed_clusters.append(new_assignments[i])
            changed_clusters = list(set(changed_clusters))

            # Update posterior probabilities for changed clusters
            for k in changed_clusters:
                log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

            sum_log_post_Z = np.sum(log_post_Z)

            # Update maximum posterior if current one is higher
            if sum_log_post_Z > self.log_max_post:
                self.log_max_post = sum_log_post_Z
                self.z_map = self.clusters.assignments.copy()
                self.iter_map = i_iter + 1

            # Check for convergence
            if sum_log_post_Z != ass_posterior:
                same_posterior_count = 0
                ass_posterior = sum_log_post_Z
            else:
                same_posterior_count += 1

            # Print current iteration state
            print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{sum_log_post_Z}", end='\r')
            
            if same_posterior_count > 4:  # Stop if the posterior hasn't changed for 5 iterations
                break
            print(f"{i_iter}/{n_iter}               ", end='\r')

        # Calculate Bayesian Information Criterion (BIC)
        self.BIC = self.clusters.K * (2 * D) * np.log(N) - (2 * self.log_max_post)

        # Print final state of the run
        print(f"\nRun: {run_id + 1}, Seed: {self.seed}, K:{len(set(self.z_map))}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")

# Main function to execute the model
if __name__ == "__main__":
    model_start_time = time.perf_counter()  # Record the start time of the model

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing Gaussian mixture data")
    parser.add_argument("-k", required=True, type=int, help="Known number of clusters or maximum number of clusters")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs with different initial assignments")
    parser.add_argument("-known", required=False, action='store_true', help="Add this flag if the number of clusters is known")
    parser.add_argument("-seed", required=False, type=int, help="Set a seed value")

    args = parser.parse_args()

    global_seed = np.random.randint(1, 2**31 - 1) if args.seed is None else args.seed
    np.random.seed(global_seed)  # Set global random seed

    ##################################  Extract data ##################################
    X = []
    dataFile = args.f
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        X.append(np.array([float(i) for i in line.strip().split(',')]))
    X = np.array(X)
    
    N = len(X)  # Number of data points
    D = len(X[0])  # Dimensionality of data

    # Model parameters
    K_max_BIC = args.k
    n_iter = 50 if args.i is None else args.i
    training_runs = 1 if args.r is None else args.r

    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, D: {D}, K: {K_max_BIC}, Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set hyper-parameters  ##################################
    alpha = 1.0 
    m_0 = np.zeros(D)
    k_0 = 0.03 
    v_0 = D + 3
    S_0 = 0.3 * v_0 * np.eye(D)
   
    # Store hyper-parameters in an object
    prior = utils.NIchi2(m_0, k_0, v_0, S_0)

    ################################## Model ##################################
    seed_l = np.random.randint(1, 2**31 - 1, training_runs)
    print(f"Total training runs: {training_runs}")

    max_post = -1 * np.inf
    least_BIC = -1 * np.inf
    for i in range(training_runs):
        seed = seed_l[i]
        print(f"\nRun:  {i + 1}, seed: {seed}")
        np.random.seed(seed)
        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)
        bayesgmm = bayesGMM(X, prior, alpha, seed, assignments=starting_assignments)
        bayesgmm.gibbs_sampler(n_iter, i)
        
        if bayesgmm.BIC > least_BIC:
            least_BIC = bayesgmm.BIC
            best_bayesgmm = bayesgmm
          
    ##################################  Model results ##################################
    z_pred_map = best_bayesgmm.z_map
    predicted_K = len(set(z_pred_map))

    print(f"\nBest Model:\nlogmax posterior: {best_bayesgmm.log_max_post}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_bayesgmm.run_id + 1} iteration: {best_bayesgmm.iter_map}")

    # Prepare predicted parameters for output
    mu_pred = []
    sigma_pred = []

    preds = {
        "mu": mu_pred,
        "sigma": np.array(sigma_pred),
        "z": z_pred_map,
        "time": time.perf_counter() - model_start_time,
        "z_last_iter": best_bayesgmm.clusters.assignments
    }

    ##################################  Save results ##################################
    outDir = "outputs_result" if args.o is None else args.o

    # Create output directory if it doesn't exist
    if outDir not in os.listdir():
        os.mkdir(outDir)
    
    outputFileName = f"{dataFilename}"    
    outputFilePath = f"{outDir}/{outputFileName}.txt"

    # Write results to a text file
    with open(outputFilePath, "w") as wFile:
        wFile.write(f"N: {N}\n")
        wFile.write(f"D: {D}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {bayesgmm.seed}\n")
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

    # Save results as a pickle file for further analysis
    outputFile = open(f"{outDir}/{outputFileName}.predParamPickle", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

    print(f"The encoded results are saved in: {outDir}/{outputFileName}.predParamPickle\n")
    print(f"The readable results are saved in: {outputFilePath}\n")