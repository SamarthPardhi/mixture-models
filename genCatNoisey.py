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

    parser.add_argument("-alpha", required=False, type=float, help="True alpha for dirichlet prior over mixing probabilities")
    
    parser.add_argument("-M", required=True, type=int, help="number of categories")
    parser.add_argument("-a", required=True, type=int, help="prior Gamma lower bound (x100)")
    parser.add_argument("-b", required=True, type=int, help="prior Gamma upper bound (x100)")

    parser.add_argument("-local", required=False, action="store_true", help="Add this flag to add local noise")
    parser.add_argument("-nd", required=True, type=int, help="Number of noisey dimension")
    
    parser.add_argument("-seed", required=False, type=int, help="Set a seed value [should not be same as you are putting in learn file]")

    args = parser.parse_args()

    seed = np.random.randint(1, 2**16 - 1) if args.seed == None else args.seed
    n = args.n 
    d = args.d
    k = args.k
    outDir = "dataCatNoisey" if args.o == None else args.o
    n_files = 1 if args.nf == None else args.nf
    isLOCAL = False if args.local == None else args.local
    nd = args.nd

    M = args.M
    a = args.a
    b = args.b

    alpha_true = 1.0 if args.alpha == None else args.alpha

    if not outDir in os.listdir():
        os.system(f"mkdir {outDir}")

    np.random.seed(seed)

    for i_file in range(n_files) :

        mixing_probabilities = np.random.dirichlet([alpha_true]*k, size=1)[0]
        z_true = np.random.choice(k, p=mixing_probabilities, size=1000)

        X = np.zeros((n, d), int)
        categorical_Ms = M + np.zeros(d, int)
        alpha_Ks = np.zeros((d, k), float)
        for i in range(d):

            # alpha_K = np.array([1, 1])
            alpha_K = 0.01* np.random.randint(a, b, k)
            alpha_Ks[i, :] = alpha_K

            categorical_M = categorical_Ms[i]

            categorical_data = np.zeros(n, int)
            for clust in range(k):
                dirichletSample_prob = np.random.dirichlet(alpha_K[clust] + np.zeros(categorical_M), size=1)[0]
                categorical_data[np.where(z_true == clust)] = np.random.choice(categorical_M, p = dirichletSample_prob, size=z_true.tolist().count(clust))
            X[:, i] = categorical_data

        filename = f"{seed}_{n}_{k}_{d}_{a}_{b}"

        features = np.ones((k, d), int)

        if not isLOCAL:
            features[:, -nd:] = np.zeros((k, nd), int)
        else:
            for j in range(nd):
                rand_cluster = np.random.randint(0, 2, k)
                features[:, j] = rand_cluster

        X_mean = np.mean(X)
        for k in range(k):
            for j in range(d):
                if features[k][j] == 0:
                    for i in range(n):
                        if z_true[i] == k :
                            X[i, j] = np.random.randint(int(5*X_mean), int(5*X_mean) + 2)

        if isLOCAL:
            newfilename = filename + f"_{nd}_local_{i_file}"
        else:
            newfilename = filename + f"_{nd}_global_{i_file}"


        wfile = open(f"{outDir}/{newfilename}.csv", "w")

        for da in X:
            data_vec = [str(i) for i in da]
            wfile.write(",".join(data_vec)+'\n')

        wfile.flush()
        wfile.close()

        wfile_f = open(f"{outDir}/{newfilename}.features", "w")

        for ff in features:
            features_vec = [str(i) for i in ff]
            wfile_f.write(",".join(features_vec)+'\n')

        wfile_z = open(f"{outDir}/{newfilename}.labels", "w")
        
        for z_i in z_true:
            wfile_z.write(f"{z_i}\n")

        print("dir/seed_n_k_d_a_b_nd_isLOCAL_file")
        print(f"{outDir}/{newfilename}")