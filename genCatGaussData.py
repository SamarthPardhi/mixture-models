import argparse
import numpy as np
import pickle
from scipy.stats import invgamma
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", required=True, type=int, help="Number of data points")
    parser.add_argument("-k", required=True, type=int, help="Number of clusters")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-nf", required=False, type=int, help="Number of files to be generated")

    parser.add_argument("-gd", required=True, type=int, help="number of gaussian dimensions")
    # parser.add_argument("-vM", required=False, type=float, help="Sigma scale mean in inverse gamma")
    # parser.add_argument("-vV", required=False, type=float, help="Sigma scale variance in inverse gamma")
    # parser.add_argument("-mB", required=False, type=float, help="Mean separations bound")
    parser.add_argument("-g", required=True, type=int, help="Maximum global averlap in percentage")

    parser.add_argument("-cd", required=True, type=int, help="number of categorical dimensions")
    parser.add_argument("-M", required=True, type=int, help="number of categories")
    parser.add_argument("-a", required=True, type=float, help="prior Alpha lower bound")
    parser.add_argument("-b", required=True, type=float, help="number of categories")

    parser.add_argument("-show", required=False, action='store_true', help="Add this flag if you want to display the heatmap")
    parser.add_argument("-cust", required=False, action='store_true', help="Add this flag for true file containing true labels")
    parser.add_argument("-seed", required=False, type=int, help="set a seed value [should not be same as you are putting in learn file]")

    args = parser.parse_args()

    n_file = 1 if args.nf == None else args.nf
    seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed 
    np.random.seed(seed)

    print("Seed: %s" % seed)

    N = args.n
    K_true = args.k

    outDir = 'dataCatGaussRdata' if args.o == None else args.o

    cD = args.cd 
    M = args.M
    a = args.a
    b = args.b

    gD = args.gd 
    goM = args.g

    for i_file in range(n_file):        

        filename_temp = f"outGaussRData/{N}_{K_true}_{gD}_{seed}_{goM}"

        os.system(f"Rscript genGaussData.R {N} {K_true} {gD} {seed} {goM} outGaussRData")

        X = []
        file = open(f"{filename_temp}.csv", "r")
        for line in file:
            X.append(np.array([float(i) for i in line.strip().split(',')]))
        X = np.array(X)

        z_true = np.array([int(line.strip()) for line in open(f"{filename_temp}.labels", "r")])

        C = np.zeros((N, cD))
        categorical_Ms = M + np.zeros(cD, int)
        alpha_Ks = np.zeros((cD, K_true), float)
        for i in range(cD):

            # alpha_K = np.array([1, 1])
            alpha_K = 0.001* np.random.randint(1000*a, 1000*b, K_true)
            alpha_Ks[i, :] = alpha_K

            categorical_M = categorical_Ms[i]

            categorical_data = np.zeros(N, int)
            for k in range(K_true):
                dirichletSample_prob = np.random.dirichlet(alpha_K[k] + np.zeros(categorical_M), size=1)[0]
                categorical_data[np.where(z_true == k)] = np.random.choice(categorical_M, p = dirichletSample_prob, size=z_true.tolist().count(k))
            C[:, i] = categorical_data

        filename = f"{seed}_{N}_{K_true}_{gD}_{cD}_{goM}_{a}_{b}_{i_file}"

        with open(f'{outDir}/{filename}.gauss.csv', 'w') as f1:
            for data_vec in X:
                data_vec = [str(i) for i in data_vec]
                f1.write(",".join(data_vec)+'\n')

        with open(f'{outDir}/{filename}.cat.csv', 'w') as f1:
            for data_vec in C:
                data_vec = [str(i) for i in data_vec]
                f1.write(",".join(data_vec)+'\n')

        with open(f'{outDir}/{filename}.csv', 'w') as f1:
            for i in range(N):
                data_vec = [str(j) for j in X[i]]
                data_vec += [str(j) for j in C[i]]
                f1.write(",".join(data_vec)+'\n')

        with open(f"{outDir}/{filename}.labels", "w") as f2:
            for z_i in z_true:
                f2.write(f"{z_i}\n")

        params_true = {
            "filename":filename,
            "seed": seed,
            "z": z_true,
            "goM":goM,
            "Ms": categorical_Ms,
            "a":a,
            "b":b,
            "gammas":alpha_Ks
        } 

        with open(f"{outDir}/{filename}.p", 'wb') as f2:
            pickle.dump(params_true, f2, pickle.HIGHEST_PROTOCOL)

        print(f"The gaussian features are stored in {outDir}/{filename}.gauss.csv.\nThe categorical features are stored in {outDir}/{filename}.cat.csv.\n Data with both features are stored in {outDir}/{filename}.csv \n The seed used is {seed}.")
        print(f"The created data parameters is stored in {outDir}/{filename}.p")