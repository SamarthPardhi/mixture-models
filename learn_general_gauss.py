import numpy as np
from cluster_stats_new import gaussianClusters
import utils
import time
import argparse
import pickle
import os
import time
from sklearn.metrics.cluster import adjusted_rand_score

class bayesGMM():
    def __init__(self, X, prior, alpha, seed, assignments):
        self.alpha = alpha
        N, D = X.shape
        
        K = len(set(assignments))
        self.K_max = K

        self.seed = seed

        self.clusters = gaussianClusters(X, prior, alpha, K, assignments)
        self.z_map = assignments
        self.iter_map = 0
        self.log_max_post = -1*np.inf
        self.BIC = 0.
        self.run_id = -1

    def gibbs_sampler(self, n_iter, run_id):
        self.run_id = run_id
        
        np.random.seed(self.seed)

        same_posterior_count = 0
        ass_posterior = 0

        log_post_Z = np.zeros(self.K_max)
        for k in range(self.K_max):
            log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

        # params_true = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))
        # print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{np.sum(log_post_Z)}, ARI: {round(adjusted_rand_score(params_true['z'], self.clusters.assignments), 3)}")
        print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{np.sum(log_post_Z)}", end="\r")

        for i_iter in range(n_iter):
            old_assignments = self.clusters.assignments.copy()

            for i in range(self.clusters.N):
                
                k_old = self.clusters.assignments[i]
                K_old = self.clusters.K
                stats_old = self.clusters.cache_cluster_stats(k_old)

                self.clusters.del_assignment(i)
                
                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(N + self.alpha - 1)

                log_prob_x_i = self.clusters.log_post_pred(i)

                log_prob_z_k = log_prob_z_k_alpha + log_prob_x_i

                # k = utils.sample_gumbel(log_prob_z_k)
                # k = utils.sample(log_prob_z_k)
                k = utils.sample_numpy_gumbel(log_prob_z_k)

                if k >= self.clusters.K:
                    k = self.clusters.K
            
                if k==k_old and self.clusters.K == K_old:
                    self.clusters.restore_cluster_stats(k_old, *stats_old)
                    self.clusters.assignments[i] = k_old

                else:
                    self.clusters.add_assignment(i,k)


            new_assignments = self.clusters.assignments
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(N):
                if not assignments_change[i]:
                    changed_clusters.append(old_assignments[i])
                    changed_clusters.append(new_assignments[i])
            changed_clusters = list(set(changed_clusters))

            for k in changed_clusters:
                log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

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

            # params_true = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))
            # print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{sum_log_post_Z}, ARI: {round(adjusted_rand_score(params_true['z'], self.clusters.assignments), 3)}")
            # print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{sum_log_post_Z}, ARI: {adjusted_rand_score(params_true['z'], self.clusters.assignments)}, ARI max post: {round(adjusted_rand_score(params_true['z'], self.z_map), 2)}")
            print(f"run: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{sum_log_post_Z}", end='\r')
            
            if same_posterior_count > 4:
                break
            print(f"{i_iter}/{n_iter}               ",end='\r')

        self.BIC = self.clusters.K*(2*D) * np.log(N) - (2 * self.log_max_post)

        print(f"\nRun: {run_id + 1}, Seed: {self.seed}, K:{len(set(self.z_map))}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")


if __name__ == "__main__":
    model_start_time = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing gauusian mixture data")
    parser.add_argument("-k", required=True, type=int, help="Known K and if it's unknown Maximum number of clusters (Or your guess that the number of clusters can't be more than that)")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs to run with different initial assignments")
    parser.add_argument("-known", required=False, action='store_true', help="add this flag if the number of clusters are known")
    parser.add_argument("-seed", required=False, type=int, help="set a seed value")

    args = parser.parse_args()

    global_seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed
    # seed = 82   
    np.random.seed(global_seed) # should not be same as in learn file
    # np.random.seed(np.random.randint(1, 2**31 - 1))

    ##################################  Extract data ##################################

    X = []
    dataFile = args.f
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        X.append(np.array([float(i) for i in line.strip().split(',')]))
    X = np.array(X)
    
    N = len(X)
    D = len(X[0])

    # model parameters
    K_max_BIC = args.k
    n_iter = 50 if args.i == None else args.i
    training_runs = 1 if args.r == None else args.r

    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, D: {D}, K: {K_max_BIC}, Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set hyper-parameters  ################################## (can we look at the data to set hyperparameters?)
    alpha = 1.0 
    m_0 = np.zeros(D)
    k_0 = 0.03 
    v_0 = D + 3
    S_0 = 0.3*v_0*np.eye(D)
   
    # just storing it in an object
    prior = utils.NIchi2(m_0, k_0, v_0, S_0)

    ################################## Model ##################################
    seed_l = np.random.randint(1, 2**31 -1, training_runs)
    # seed_l = np.arange(1, training_runs + 1)
    print(f"Total training runs: {training_runs}")

    max_post = -1*np.inf
    least_BIC = -1*np.inf
    for i in range(training_runs):
        seed = seed_l[i]
        print(f"\nRun:  {i+1}, seed: {seed}")
        np.random.seed(seed)
        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)
        # starting_assignments = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))['z']
        bayesgmm = bayesGMM(X, prior, alpha, seed, assignments = starting_assignments)
        bayesgmm.gibbs_sampler(n_iter, i)
        
        if bayesgmm.BIC > least_BIC:
            least_BIC = bayesgmm.BIC
            best_bayesgmm =bayesgmm
          
    ##################################  Model results ##################################

    z_pred_map = best_bayesgmm.z_map
    predicted_K = len(set(z_pred_map))

    print(f"\nBest Model:\nlogmax posterior: {best_bayesgmm.log_max_post}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_bayesgmm.run_id + 1} iteration: {best_bayesgmm.iter_map}")

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

    outDir = "outputs_result" if args.o == None else args.o

    if outDir not in os.listdir():
        os.mkdir(outDir)
    
    outputFileName = f"{dataFilename}"    
    outputFilePath = f"{outDir}/{outputFileName}.txt"

    with open(outputFilePath, "w") as wFile:
        wFile.write(f"N: {N}\n")
        wFile.write(f"D: {D}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {bayesgmm.seed}\n")
        wFile.write(f"Iterations: {n_iter}\n")
        wFile.write(f"alpha: {alpha}\n")
        wFile.write(f"time: {time.perf_counter() - model_start_time}")

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

    outputFile = open(f"{outDir}/{outputFileName}.predParamPickle", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

    print(f"The encoded results are saved in: {outDir}/{outputFileName}.predParamPickle\n")
    print(f"The readable results are saved in: {outputFilePath}\n")