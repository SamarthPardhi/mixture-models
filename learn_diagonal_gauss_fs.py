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
    def __init__(self, X, prior, alpha, assignments, FS,  features = []):
        self.FS = FS

        self.trueZ = []
        self.alpha = alpha

        K = len(set(assignments))
        self.K_max = K

        self.clusters = gaussianClustersDiagFS(X, prior, alpha, K, assignments.copy(), FS, features)
        self.z_map = assignments
        self.iter_map = 0
        self.log_max_post = -1*np.inf
        self.BIC = 0.
        self.run_id = -1


    def gibbs_sampler(self, n_iter, run_id, toPrint=True, savePosterior=False, trueAssignments=[], greedyRun = False):
        if toPrint:
            print(f"Initial features:")
            print(self.clusters.features)

        if len(trueAssignments) > 0:
            self.trueZ = trueAssignments

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

                if greedyRun == True:                                
                    old_post_prob = 0
                    for k in range(self.K_max):
                        old_post_prob += self.clusters.get_posterior_probability_Z_k(k)

                old_assignments_i = self.clusters.assignments.copy()

                k_old = self.clusters.assignments[i]
                K_old = self.clusters.K
                stats_old = self.clusters.cache_cluster_stats(k_old)
                k_counts_old = self.clusters.counts[k_old]

                self.clusters.del_assignment(i)                

                log_prob_z_k_alpha = np.log(self.clusters.counts + self.alpha / self.clusters.K_max ) - np.log(self.clusters.N + self.alpha - 1)

                log_prob_x_i = self.clusters.log_post_pred0(i)
                log_prob_z_k = log_prob_z_k_alpha + log_prob_x_i

                changed_ = []

                k_new = np.argmax(log_prob_z_k + np.random.gumbel(0, 1, len(log_prob_z_k)))

                if k_new >= self.clusters.K:
                    k_new = self.clusters.K

                if k_new == k_old and self.clusters.K == K_old:
                    self.clusters.restore_cluster_stats(k_old, *stats_old)
                    self.clusters.assignments[i] = k_old

                else:
                    self.clusters.add_assignment(i, k_new)
                    changed_ += [k_new, k_old]

                if k_counts_old == 1:
                    changed_.append(K_old - 1)
                
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

                # if self.FS:

                #     log_prob_unimp = self.clusters.log_prob_unimp_marginal()

                #     log_prob_imp =  self.clusters.log_prob_imp_marginal()

                #     for k in range(self.clusters.K):
                #         for j in range(self.clusters.D):
                #             self.clusters.features[k][j] = np.argmax(np.array([log_prob_unimp[k][j], log_prob_imp[k][j]]) + np.random.gumbel(0, 1, 2))


                if savePosterior:

                    if len(changed_) > 4:
                        print(sorted(changed_))
                        print([k_old, k_new, K_old, self.clusters.K])
                        breakpoint()

                    for k in changed_:
                        log_post_Z[k] =  self.clusters.get_posterior_probability_Z_k(k)
                    
                    posteriorList.append(np.sum(log_post_Z))
                    ARI_list.append(round(adjusted_rand_score(self.trueZ, self.clusters.assignments), 3))    

            ####### Feature Selection ###########
            
            if self.FS:

                lamb = 200
                log_prob_unimp = self.clusters.log_prob_unimp_marginal(lamb)
                log_prob_imp =  self.clusters.log_prob_imp_marginal(lamb)

                for k in range(self.clusters.K):
                    for j in range(self.clusters.D):
                        if log_prob_unimp[k][j] == log_prob_imp[k][j]:
                            self.clusters.features[k][j] = 1
                        else:
                            self.clusters.features[k][j] = np.argmax(np.array([log_prob_unimp[k][j], log_prob_imp[k][j]]) + np.random.gumbel(0, 1, 2))
                
                # print(self.clusters.features)
                        
                # print(log_prob_unimp[0, :])
                # print(log_prob_imp[0, :])
                # breakpoint()

            new_assignments = self.clusters.assignments
            assignments_change = old_assignments == new_assignments
            changed_clusters = []
            for i in range(self.clusters.N):
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


            if toPrint:

                if len(self.trueZ) != 0:
                    print(f"\nrun: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{round(sum_log_post_Z, 3)}, ARI: {adjusted_rand_score(self.trueZ, self.clusters.assignments.copy())}")
                    if self.FS:
                        print("features:\n", self.clusters.features)

                else:
                    print(f"\nrun: {run_id + 1}, iteration:{i_iter + 1}, K:{self.clusters.K}, posterior:{round(sum_log_post_Z, 3)}")
                    # print("features:\n", self.clusters.features)

            # breakpoint()

            if same_posterior_count > 3:
                break

            print(f"{i_iter}/{n_iter}               ",end='\r')

        self.BIC = self.clusters.get_no_free_param() - 2 * self.log_max_post
        print(f"\nRun: {run_id + 1}, K:{len(set(self.z_map))}, BIC: {self.BIC}, logmax post: {self.log_max_post}, max_post_iter: {self.iter_map}")
        
        print("Final features:")
        print(self.clusters.features)
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

    parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing gauusian mixture data")
    parser.add_argument("-k", required=False, type=int, help="Known K and if it's unknown Maximum number of clusters (Or your guess that the number of clusters can't be more than that)")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
    parser.add_argument("-r", required=False, type=int, help="Number of training runs to run with different initial assignments")
    parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the true parameters file (non-pickle file)")
    parser.add_argument("-p", required=False, action="store_true", help="Will print results while Gibbs sampling")
    parser.add_argument("-seed", required=False, type=int, help="Set a seed value")
    parser.add_argument("-fs", required=False, action="store_true", help="Add this flag if you want to do feature selection")
    parser.add_argument("-fd", required=False, type=argparse.FileType('r'), help="Path to the file containing true feature importance")
    

    args = parser.parse_args()

    global_seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed
    np.random.seed(global_seed)

    ##################################  Extract data ##################################

    X = []
    dataFile = args.f
    dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]
    for line in dataFile:
        X.append(np.array([float(i) for i in line.strip().split(',')]))
    X = np.array(X)

    # X = X[:, 4:]
    
    N = len(X)
    D = len(X[0])
    
    # model parameters
    K_max_BIC = 50 if args.k == None else args.k
    n_iter = 50 if args.i == None else args.i

    training_runs = 1 if args.r == None else args.r

    print(f"\nRunning {os.path.basename(__file__)} on {dataFilename} with global seed: {global_seed}")
    print(f"N: {N}, D: {D}, K: {K_max_BIC}, Iterations: {n_iter}, Global seed: {global_seed}\n")

    ################################## Set hyper-parameters  ################################## (can we look at the data to set hyperparameters?)
    alpha = 1.0 
    m_0 = np.zeros(D)
    k_0 = 0.1 
    v_0 = 1
    S_0 = v_0 * np.ones(D)

    #define MU0 0
    #define KAPPA0 0.1
    #define ALPHA0 0.1
    #define BETA0 1
   
    # just storing it in an object
    prior = utils.NIchi2(m_0, k_0, v_0, S_0)

    ################################## Model ##################################
    
    if args.p:
        toDisplay = True
    else:
        toDisplay = False

    if args.fs:
        FS = True
    else:
        FS = False

    true_features = []
    if args.fd:
        featureFile = args.fd
        for line in featureFile:
            true_features.append(np.array([int(i) for i in line.strip().split(',')]))
        true_features = np.array(true_features)

    # true_features = np.concatenate([np.zeros((K_max_BIC, 4), int), np.ones((K_max_BIC, 6), int)], axis=1)

    print(f"Total training runs: {training_runs}")

    trueFile = args.t
    if trueFile:
        trueAssignments = np.array([int(line.strip()) for line in trueFile])
    else:
        trueAssignments = []

    max_post = -1*np.inf
    least_BIC = -1*np.inf
    for i in range(training_runs):
        print(f"\nRun:  {i+1}")

        starting_assignments = []
        while len(set(starting_assignments)) != K_max_BIC:
            starting_assignments = np.random.randint(0, K_max_BIC, N)

        # starting_assignments = trueAssignments

        # params_true = pickle.load(open("../data_n1000_d10_k10_m2.0_c2.1_catD0_catM4_seed1616.trueParamPickle", "rb"))
        # starting_assignments = params_true['z']
        # starting_assignments = params_true = np.array(json.load(open("../Z_true.json", "rb"))['z'])
        # starting_assignments = pickle.load(open("../data_n1000_d0_k5_m2.1_c2.1_catD1_catM4_seed23.trueParamPickle", "rb"))['z']
        
        bayesgmm = bayesGMM_FS(X, prior, alpha, starting_assignments, FS, features=true_features)
        bayesgmm.gibbs_sampler(n_iter, i, trueAssignments=trueAssignments, toPrint=toDisplay, greedyRun=False, savePosterior=False)
        
        if bayesgmm.BIC > least_BIC:
            least_BIC = bayesgmm.BIC
            best_bayesgmm = bayesgmm

    ##################################  Model results ##################################

    z_pred_map = best_bayesgmm.z_map
    predicted_K = len(set(z_pred_map))

    print(f"\nBest Model:\nlogmax posterior: {best_bayesgmm.log_max_post}\nBIC: {best_bayesgmm.BIC}\nPredicted K (MAP): {predicted_K}\nmax post run: {best_bayesgmm.run_id + 1} iteration: {best_bayesgmm.iter_map}")
    print(f"Time: {time.perf_counter() - model_start_time}")
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

    if args.o == None:
        if args.fs:
            outDir = "outGaussFS"
        else:
            outDir = "outGaussNoisey"
    else:
        outDir = args.o

    outputFileName = f"{dataFilename}"    
    outputFilePath = f"{outDir}/{outputFileName}.txt"

    with open(outputFilePath, "w") as wFile:
        wFile.write(f"N: {N}\n")
        wFile.write(f"D: {D}\n")
        wFile.write(f"K: {predicted_K}\n\n")
        wFile.write(f"Seed: {global_seed}\n")
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

    outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
    pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)
    
    with open(f"{outDir}/{outputFileName}.labels", "w") as ff:
        for z_i in z_pred_map:
            ff.write(f"{z_i}\n")

    with open(f'{outDir}/{outputFileName}.features', 'w') as ff1:
        for data_vec in best_bayesgmm.clusters.features:
            data_vec = [str(i) for i in data_vec]
            ff1.write(",".join(data_vec)+'\n')

    print(f"The encoded results are saved in: {outDir}/{outputFileName}.p\n")
    print(f"The readable feature importance are saved in: {outDir}/{outputFileName}.features\n")
    print(f"The readable results are saved in: {outputFilePath}\n")