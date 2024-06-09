import argparse
import numpy as np
import pickle
from scipy.stats import invgamma
import os


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", required=True, type=int, help="Number of data points")
    parser.add_argument("-d", required=True, type=int, help="number of gaussian dimensions")
    parser.add_argument("-k", required=True, type=int, help="Number of clusters")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-nf", required=False, type=int, help="Number of files to be generated")
    parser.add_argument("-goM", required=True, type=int, help="generalised overlap of Maitra")
    
    parser.add_argument("-local", required=False, action="store_true", help="Add this flag if you want to do feature selection")
    parser.add_argument("-nd", required=True, type=int, help="number of noisey dimension")
    
    parser.add_argument("-seed", required=False, type=int, help="set a seed value [should not be same as you are putting in learn file]")

    args = parser.parse_args()

    seed = np.random.randint(1, 2**16 - 1) if args.seed == None else args.seed
    n = args.n 
    d = args.d
    k = args.k
    outDir = "dataGaussNoisey" if args.o == None else args.o
    n_files = 1 if args.nf == None else args.nf
    goM = args.goM
    isLOCAL = False if args.local == None else args.local
    nd = args.nd

    if not outDir in os.listdir():
        os.system(f"mkdir {outDir}")

    np.random.seed(seed)

    for i_file in range(n_files) :
        print(f"Rscript genGaussData.R {n} {k} {d} {seed} {goM} {outDir}")
        os.system(f"Rscript genGaussData.R {n} {k} {d} {seed} {goM} {outDir}")
        filename = f"{outDir}/{n}_{k}_{d}_{seed}_{goM}"

        X = []
        dataFile = open(f"{filename}.csv", "r")
        for line in dataFile:
            X.append(np.array([float(i) for i in line.strip().split(',')]))
        X = np.array(X)

        z_true = [int(i.strip()) for i in open(f"{filename}.labels", "r")]
        
        features = np.ones((k, d), int)

        if not isLOCAL:
            features[:, -nd:] = np.zeros((k, nd), int)
        else:
            for j in range(nd):
                rand_cluster = np.random.randint(0, 2, k)
                features[:, j] = rand_cluster


        X_mean = np.mean(X)
        X_var = np.var(X)
        for k in range(k):
            for j in range(d):
                if features[k][j] == 0:
                    for i in range(n):
                        if z_true[i] == k :
                            X[i, j] = np.random.normal(10*X_mean, 0.1*X_var)


        if isLOCAL:
            newfilename = filename + f"_{nd}_local_{i_file}"
        else:
            newfilename = filename + f"_{nd}_global_{i_file}"


        wfile = open(f"{newfilename}.csv", "w")

        for da in X:
            data_vec = [str(i) for i in da]
            wfile.write(",".join(data_vec)+'\n')

        wfile.flush()
        wfile.close()

        wfile_f = open(f"{newfilename}.features", "w")

        for ff in features:
            features_vec = [str(i) for i in ff]
            wfile_f.write(",".join(features_vec)+'\n')

        os.system(f"mv {filename}.labels {newfilename}.labels")
        print(newfilename)

