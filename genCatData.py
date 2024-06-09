import argparse
import numpy as np
import pickle

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", required=True, type=int, help="Number of data points")
    parser.add_argument("-d", required=True, type=int, help="number of gaussian dimensions")
    parser.add_argument("-k", required=True, type=int, help="Number of clusters")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-nf", required=False, type=int, help="Number of files to be generated")
    parser.add_argument("-alpha", required=False, type=float, help="True alpha for dirichlet prior over mixing probabilities")
    
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
    D = args.d 
    K_true = args.k

    outDir = 'dataCat' if args.o == None else args.o

    alpha_true = 1.0 if args.alpha == None else args.alpha

    M = args.M
    a = args.a
    b = args.b

    for i_file in range(n_file):        
        mixing_probabilities = np.random.dirichlet([alpha_true]*K_true, size=1)[0]
        z_true = np.random.choice(K_true, p=mixing_probabilities, size=1000)

        X = np.zeros((N, D))
        categorical_Ms = M + np.zeros(D, int)
        alpha_Ks = np.zeros((D, K_true), float)
        for i in range(D):

            # alpha_K = np.array([1, 1])
            alpha_K = 0.001* np.random.randint(1000*a, 1000*b, K_true)
            alpha_Ks[i, :] = alpha_K

            categorical_M = categorical_Ms[i]

            categorical_data = np.zeros(N, int)
            for k in range(K_true):
                dirichletSample_prob = np.random.dirichlet(alpha_K[k] + np.zeros(categorical_M), size=1)[0]
                categorical_data[np.where(z_true == k)] = np.random.choice(categorical_M, p = dirichletSample_prob, size=z_true.tolist().count(k))
            X[:, i] = categorical_data

        filename = f"{seed}_{N}_{K_true}_{D}_{a}_{b}_{i_file}"
        
        with open(f'{outDir}/{filename}.csv', 'w') as f1:
            for data_vec in X:
                data_vec = [str(i) for i in data_vec]
                f1.write(",".join(data_vec)+'\n')

        with open(f"{outDir}/{filename}.labels", "w") as f2:
            for z_i in z_true:
                f2.write(f"{z_i}\n")

        params_true = {
            "filename":filename,
            "seed": seed,
            "alpha": alpha_true,
            "z": z_true,
            "Ms": categorical_Ms,
            "a":a,
            "b":b,
            "gammas":alpha_Ks
        }

        with open(f"{outDir}/{filename}.p", 'wb') as f2:
            pickle.dump(params_true, f2, pickle.HIGHEST_PROTOCOL)
        
        print(f"The created data points are stored in {outDir}/{filename}.csv. The seed used is {seed} (saved in the following pickle file)")
        print(f"The created data parameters is stored in {outDir}/{filename}.p")