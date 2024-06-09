from tracemalloc import start
import numpy as np
from cluster_stats_new import categoricalGaussianClusters
import utils
import time
import argparse
import pickle
import os
import time
from sklearn.metrics.cluster import adjusted_rand_score
import json
from scipy.special import logsumexp

class bayesCGMM():
    def __init__(self, X, C, alpha, gamma, prior, assignments, isTrueZ=0):

        if isTrueZ == 1:
            self.trueZ = assignments.copy()
        else:
            self.trueZ = []
        
        self.alpha = alpha

        K = len(set(assignments))
        self.K_max = K
        
        self.N, cD = C.shape
        self.Ms = np.zeros(cD, int)
        for d in range(cD):
            self.Ms[d] = len(set(C[d]))

        self.clusters = categoricalGaussianClusters(X, C, alpha, prior, gamma, K, assignments)
        self.z_map = assignments
        self.iter_map = 0
        self.log_max_post = -1*np.inf
        self.BIC = 0.
        self.run_id = -1

    def gibbs_sampler(self, n_iter, run_id, toPrint=True, savePosterior=False, trueAssignments=[]):

        self.trueZ = trueAssignments.copy()

        self.run_id = run_id

        posteriorList = []
        ARI_list = []
        same_posterior_count = 0
        ass_posterior = 0

        log_post_Z = np.zeros(self.K_max)
        for k in range(self.K_max):
            log_post_Z[k] = self.clusters.get_posterior_probability_Z_k(k)

        if toPrint:
            if len(self.trueZ) != 0:
                print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{round(np.sum(log_post_Z), 3)}, ARI: {round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3)}")
            else:
                print(f"run: {run_id + 1}, iteration:0, K:{self.clusters.K}, posterior:{round(np.sum(log_post_Z), 3)}")

        for i_iter in range(n_iter):
            old_assignments = self.clusters.assignments.copy()

            for i in range(self.clusters.N):
                
                k_old = self.clusters.assignments[i]
                K_old = self.clusters.K
                stats_old = self.clusters.cache_cluster_stats(k_old)

                self.clusters.del_assignment(i)
                
                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(self.clusters.N + self.alpha - 1)

                log_prob_x_i = self.clusters.log_post_pred_cat(i) + self.clusters.log_post_pred_gauss(i)
                
                log_prob_z_k = log_prob_z_k_alpha + log_prob_x_i

                # k = utils.sample(log_prob_z_k)
                # if self.clusters.K < self.K_max:
                #     log_prob_z_k = log_prob_z_k[:(self.clusters.K+1)]
                
                k = np.argmax(log_prob_z_k + np.random.gumbel(0, 1, len(log_prob_z_k)))
                
                # if self.clusters.K < self.K_max - 2:
                #     breakpoint()
                # prob_z =  np.exp(log_prob_z_k - logsumexp(log_prob_z_k))
                # k = np.random.choice(len(prob_z), p=prob_z)

                # if an empty cluster is sampled
                if k >= self.clusters.K:
                    k = self.clusters.K

                # if the same old assignment is sampled AND deleting i-th data point didn't make the cluser empty
                if k==k_old and self.clusters.K == K_old:
                    self.clusters.restore_cluster_stats(k_old, *stats_old)
                    self.clusters.assignments[i] = k_old

                else:
                    self.clusters.add_assignment(i,k)

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
                    for k in changed_:
                        log_post_Z_[k] =  self.clusters.get_posterior_probability_Z_k(k)

                    posteriorList.append(np.sum(log_post_Z_))
                    ARI_list.append(round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3))

            new_assignments = self.clusters.assignments
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(self.clusters.N):
                if not assignments_change[i]:
                    changed_clusters.append(old_assignments[i])
                    changed_clusters.append(new_assignments[i])
            changed_clusters = list(set(changed_clusters))

            for k in changed_clusters:
                log_post_Z[k] =  self.clusters.get_posterior_probability_Z_k(k)

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
    model_start_time = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument("-fc", required=True, type=argparse.FileType('r'), help="Path to the file containing gaussian features data")
    parser.add_argument("-fg", required=True, type=argparse.FileType('r'), help="Path to the file containing categorical features data")
    parser.add_argument("-k", required=True, type=int, help="Known K and if it's unknown Maximum number of clusters (Or your guess that the number of clusters can't be more than that)")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs to run with different initial assignments")
    parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the true parameters file (non-pickle file)")
    parser.add_argument("-p", required=False, action="store_true", help="Will print results while Gibbs sampling")
    parser.add_argument("-seed", required=False, type=int, help="set a seed value")

    args = parser.parse_args()

    global_seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed
    np.random.seed(global_seed)

    ##################################  Extract data ##################################

    X = []
    dataFile = args.fg
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        X.append(np.array([float(i) for i in line.strip().split(',')]))
    X = np.array(X)
    
    C = []
    dataFile = args.fc
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        C.append(np.array([int(float(i)) for i in line.strip().split(',')]))
    C = np.array(C)

    N, cD = C.shape
    gD = len(X[0])

    # model parameters
    K_max_BIC = args.k
    n_iter = 50 if args.i == None else args.i

    training_runs = 1 if args.r == None else args.r

    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, gD: {cD}, fD: {gD}, K: {K_max_BIC}, Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set Gaussian hyper-parameters  ################################## (can we look at the data to set hyperparameters?)
    alpha = 1.0 
    m_0 = np.zeros(gD)
    k_0 = 0.03 
    v_0 = gD + 3
    S_0 = 0.3*v_0*np.ones(gD)
   
    # just storing it in an object
    prior = utils.NIchi2(m_0, k_0, v_0, S_0)

    ################################## Set Categorical hyper-parameters  ################################## (can we look at the data to set hyperparameters?)
    alpha = 1.0
    gamma = 0.2

    ################################## Model ##################################

    print(f"Total training runs: {training_runs}")

    trueFile = args.t
    if trueFile:
        trueAssignments = np.array([int(line.strip()) for line in trueFile])
        # print(trueAssignments, sep=",")
        bayescgmm = bayesCGMM(X, C, alpha, gamma, prior, trueAssignments, 1)
        bayescgmm.gibbs_sampler(n_iter, -1)
    else:
        trueAssignments = []


    max_post = -1*np.inf
    least_BIC = -1*np.inf
    for i in range(training_runs):
        print(f"\nRun:  {i+1}")

        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)

        # params_true = pickle.load(open("../data_n1000_d10_k10_m2.0_c2.1_catD0_catM4_seed1616.trueParamPickle", "rb"))
        # starting_assignments = params_true['z']
        # starting_assignments = params_true = np.array(json.load(open("../Z_true.json", "rb"))['z'])
        # starting_assignments = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))['z']

        bayescgmm = bayesCGMM(X, C, alpha, gamma, prior, starting_assignments)
        bayescgmm.gibbs_sampler(n_iter, i, trueAssignments=trueAssignments)

        if bayescgmm.BIC > least_BIC:
            least_BIC = bayescgmm.BIC
            best_bayescgmm =bayescgmm

    ##################################  Model results ##################################

    z_pred_map = best_bayescgmm.z_map
    predicted_K = len(set(z_pred_map))

    print(f"\nBest Model:\nlogmax posterior: {best_bayescgmm.log_max_post}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_bayescgmm.run_id + 1} iteration: {best_bayescgmm.iter_map}")
    print(f"Time: {time.perf_counter() - model_start_time}")
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

    outDir = "outCatGauss" if args.o == None else args.o

    if outDir not in os.listdir():
        os.mkdir(outDir)
    
    outputFileName = f"{dataFilename}"    
    outputFilePath = f"{outDir}/{outputFileName}.txt"

    with open(outputFilePath, "w") as wFile:
        wFile.write(f"N: {N}\n")
        wFile.write(f"gD: {gD}\n")
        wFile.write(f"cD: {cD}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {global_seed}\n")
        wFile.write(f"Iterations: {n_iter}\n")
        wFile.write(f"alpha: {alpha}\n")
        wFile.write(f"time: {time.perf_counter() - model_start_time}")

        wFile.write(f"BIC score: {best_bayescgmm.BIC}\n")
        wFile.write(f"log max posterior: {best_bayescgmm.log_max_post}\n")
        wFile.write(f"MAP assignments: {best_bayescgmm.z_map}\n")
        wFile.write(f"Last iteration assignments: {best_bayescgmm.clusters.assignments}\n")
    
        wFile.write("m_0:")
        np.savetxt(wFile, m_0)
        wFile.write(f"k_0: {k_0}\n")
        wFile.write(f"v_0: {v_0}\n")
        wFile.write("S_0:")
        np.savetxt(wFile, S_0)

    outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)
    
    with open(f"{outDir}/{outputFileName}.labels", "w") as ff:
        for z_i in z_pred_map:
            ff.write(f"{z_i}\n")

    print(f"The encoded results are saved in: {outDir}/{outputFileName}.p\n")
    print(f"The readable results are saved in: {outputFilePath}\n")