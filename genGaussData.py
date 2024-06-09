import argparse
import numpy as np
import pickle
from scipy.stats import invgamma


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", required=True, type=int, help="Number of data points")
    parser.add_argument("-d", required=True, type=int, help="number of gaussian dimensions")
    parser.add_argument("-k", required=True, type=int, help="Number of clusters")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-nf", required=False, type=int, help="Number of files to be generated")
    parser.add_argument("-alpha", required=False, type=float, help="True alpha for dirichlet prior over mixing probabilities")
    parser.add_argument("-vM", required=False, type=float, help="Sigma scale mean in inverse gamma")
    parser.add_argument("-vV", required=False, type=float, help="Sigma scale variance in inverse gamma")
    parser.add_argument("-mB", required=False, type=float, help="Mean separations bound")

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

    outDir = 'dataGauss' if args.o == None else args.o

    alpha_true = 1.0 if args.alpha == None else args.alpha

    vM = 0 if args.vM == None else args.vM
    vV = 0 if args.vV == None else args.vV
    mB = 0 if args.mB == None else args.mB

    S_0 = np.zeros((K_true, D))
    V_0 = np.zeros((K_true, D)) + D + 3
    M_0 = np.zeros((K_true, D))
    K_0 = np.zeros((K_true, D)) + 0.07

    for d in range(D):
    
        S_0[:, d] = abs(np.random.normal(vM, vV, K_true))
        # S_0[:, d] = np.random.uniform(0, 2, K_true)
        M_0[:, d] = np.random.uniform(-1*mB, mB, K_true)

    mu_s = np.zeros((K_true, D))
    sigma_s = np.zeros((K_true, D))

    for i_file in range(n_file):        

        mixing_probabilities = np.random.dirichlet([alpha_true]*K_true, size=1)[0]
        z_true = np.random.choice(K_true, p=mixing_probabilities, size=1000)

        X = np.zeros((N, D))

        for d in range(D):

            # alpha_K = np.array([1, 1])
            gaussian_data = np.zeros(N, int)

            for k in range(K_true):
                sigma = invgamma.rvs(V_0[k, d], scale=S_0[k, d])
                mu = np.random.normal(M_0[k, d], sigma/K_0[k, d])
                mu_s[k, d] = mu
                sigma_s[k, d] = sigma

                gaussian_data[np.where(z_true == k)] = np.random.normal(mu, sigma, z_true.tolist().count(k))

            X[:, d] = gaussian_data

        filename = f"{seed}_{N}_{K_true}_{D}_{vM}_{vV}_{mB}_{i_file}"
        
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
            "alpha":alpha_true,
            "pi":mixing_probabilities,
            "z": z_true,
            "S_0":S_0,
            "M_0":M_0,
            "K_0":K_0,
            "V_0":V_0,
            "means":mu_s,
            "variances":sigma_s
        } 

        with open(f"{outDir}/{filename}.p", 'wb') as f2:
            pickle.dump(params_true, f2, pickle.HIGHEST_PROTOCOL)
        
        print(f"The created data points are stored in {outDir}/{filename}.csv. The seed used is {seed} (saved in the following pickle file)")
        print(f"The created data parameters is stored in {outDir}/{filename}.p")