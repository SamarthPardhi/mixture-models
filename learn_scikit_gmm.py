import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import argparse
import os
import pickle
import time
from sklearn.metrics.cluster import adjusted_rand_score

model_start_time = time.perf_counter()

# global_seed = 763
global_seed = np.random.randint(1, 2**31-1)
np.random.seed(global_seed)

parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing gauusian mixture data")
parser.add_argument("-k", required=False, type=int, help="Known K and if it's unknown Maximum number of clusters (Or your guess that the number of clusters can't be more than that)")
parser.add_argument("-i", required=False, type=int, help="Collapsed Gibbs sampling iterations")
parser.add_argument("-o", required=False, type=str, help="Output directory")
parser.add_argument("-r", required=False, type=int, help="Number of training runs to run with different initial assignments")
parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the file containing true labels of gauusian mixture")

parser.add_argument("-known", required=False, action='store_true', help="add this flag if the number of clusters are known")

args = parser.parse_args()

training_runs = 2 if args.r == None else args.r
max_iterations = 1 if args.i == None else args.i

X = []
dataFile = args.f
dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]

for line in dataFile:
    X.append(np.array([float(i) for i in line.strip().split(',')]))
X = np.array(X)

N = len(X)
D = len(X[0])

##########################################################
print(f"Total Training Runs: {training_runs}")

trueFile = args.t
if trueFile:
    trueAssignments = np.array([int(line.strip()) for line in trueFile])
    K_true = len(set(trueAssignments))

    K = K_true
    best_gmm = GaussianMixture(n_components=K, n_init=training_runs*max_iterations)
    best_gmm.fit(X)
    print(f"K: {K}, BIC: {best_gmm.bic(X)}")
    z_pred = best_gmm.predict(X)

else:
    minK = 2
    maxK = args.k
    print(f"Implementing BIC for k = {minK}, ..., {maxK}\n")
    lowest_bic_score = 1*np.inf
    for K in range(minK, maxK):
        gmm = GaussianMixture(n_components=K, n_init=training_runs*max_iterations)
        gmm.fit(X)
        # print(f"K: {K}, BIC: {gmm.bic(X)}", end="\r")
        print(f"K: {K}, BIC: {gmm.bic(X)}")
        if gmm.bic(X) < lowest_bic_score:
            best_gmm = gmm
            lowest_bic_score = gmm.bic(X)

        # if K == 4:
        #     preds_K = {
        #         "mu": [],
        #         "sigma": [],
        #         "z": gmm.predict(X),
        #         "time":time.perf_counter() - model_start_time,
        #     }

    z_pred = best_gmm.predict(X)
    predicted_K = len(set(z_pred))
    print(f"\nPredicted K through BIC: ", predicted_K)

preds = {
    "mu": [],
    "sigma": [],
    "z": z_pred,
    "time":time.perf_counter() - model_start_time,
    "z_last_iter": z_pred
}


##################################  Save results ##################################

outDir = "outScikitGMM" if args.o == None else args.o

if outDir not in os.listdir():
    os.mkdir(outDir)

outputFileName = f"{dataFilename}"
outputFile = open(f"{outDir}/{outputFileName}.p", "wb")

pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

print(f"ARI: {adjusted_rand_score(preds['z'], trueAssignments)}")

print(f"The encoded results are saved in: {outDir}/{outputFileName}.p\n")

outputFileName = f"{dataFilename}"
outputFile = open(f"{outDir}/{outputFileName}.p", "wb")

pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)
