import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import argparse
import os
import pickle
import time
from sklearn.metrics.cluster import adjusted_rand_score

# Record the start time of the model
model_start_time = time.perf_counter()

# Set a global random seed for reproducibility
global_seed = np.random.randint(1, 2**31-1)
np.random.seed(global_seed)

# Set up argument parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Path to the file containing Gaussian mixture data")
parser.add_argument("-k", required=False, type=int, help="Known K or maximum number of clusters if unknown")
parser.add_argument("-i", required=False, type=int, help="Number of Gibbs sampling iterations")
parser.add_argument("-o", required=False, type=str, help="Output directory")
parser.add_argument("-r", required=False, type=int, help="Number of training runs with different initial assignments")
parser.add_argument("-t", required=False, type=argparse.FileType('r'), help="Path to the file containing true labels of Gaussian mixture")
parser.add_argument("-known", required=False, action='store_true', help="Flag to indicate if the number of clusters is known")

# Parse the arguments
args = parser.parse_args()

# Set the number of training runs and maximum iterations
training_runs = 2 if args.r is None else args.r
max_iterations = 1 if args.i is None else args.i

# Load the dataset from the provided file
X = []
dataFile = args.f
dataFilename = os.path.splitext(os.path.basename(dataFile.name))[0]

for line in dataFile:
    X.append(np.array([float(i) for i in line.strip().split(',')]))
X = np.array(X)

N = len(X)
D = len(X[0])

##########################################################
# Print the total number of training runs
print(f"Total Training Runs: {training_runs}")

# Check if true labels file is provided
trueFile = args.t
if trueFile:
    # Load the true assignments if available
    trueAssignments = np.array([int(line.strip()) for line in trueFile])
    K_true = len(set(trueAssignments))

    # Use the true number of clusters for Gaussian Mixture Model
    K = K_true
    best_gmm = GaussianMixture(n_components=K, n_init=training_runs * max_iterations)
    best_gmm.fit(X)
    print(f"K: {K}, BIC: {best_gmm.bic(X)}")
    z_pred = best_gmm.predict(X)
else:
    # If true labels are not provided, use BIC to determine the optimal number of clusters
    minK = 2
    maxK = args.k
    print(f"Implementing BIC for k = {minK}, ..., {maxK}\n")
    lowest_bic_score = np.inf
    for K in range(minK, maxK):
        gmm = GaussianMixture(n_components=K, n_init=training_runs * max_iterations)
        gmm.fit(X)
        print(f"K: {K}, BIC: {gmm.bic(X)}")
        if gmm.bic(X) < lowest_bic_score:
            best_gmm = gmm
            lowest_bic_score = gmm.bic(X)

    z_pred = best_gmm.predict(X)
    predicted_K = len(set(z_pred))
    print(f"\nPredicted K through BIC: ", predicted_K)

# Prepare the prediction results
preds = {
    "mu": [],
    "sigma": [],
    "z": z_pred,
    "time": time.perf_counter() - model_start_time,
    "z_last_iter": z_pred
}

##################################  Save results ##################################
# Define output directory
outDir = "outScikitGMM" if args.o is None else args.o

# Create the output directory if it doesn't exist
if outDir not in os.listdir():
    os.mkdir(outDir)

# Save the results as a pickle file
outputFileName = f"{dataFilename}"
outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)

# Print the Adjusted Rand Index (ARI) if true labels are available
if trueFile:
    print(f"ARI: {adjusted_rand_score(preds['z'], trueAssignments)}")

# Print the location of the saved results
print(f"The encoded results are saved in: {outDir}/{outputFileName}.p\n")

# Save the results again (redundant code, can be removed)
outputFileName = f"{dataFilename}"
outputFile = open(f"{outDir}/{outputFileName}.p", "wb")
pickle.dump(preds, outputFile, pickle.HIGHEST_PROTOCOL)
