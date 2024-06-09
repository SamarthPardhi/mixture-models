import numpy as np
import argparse
import pickle
import os
from scipy.stats import invwishart
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", required=True, type=int, help="Number of data points")
    parser.add_argument("-d", required=True, type=int, help="number of gaussian dimensions")
    parser.add_argument("-k", required=True, type=int, help="Number of clusters")
    parser.add_argument("-o", required=False, type=str, help="Output directory")
    parser.add_argument("-m", required=False, type=float, help="mu scale")
    parser.add_argument("-c", required=False, type=float, help="covariance scale")
    parser.add_argument("-f", required=False, type=str, help="output file name format")
    parser.add_argument("-cat", required=False, type=int, help="number of categorical data dimensions")
    parser.add_argument("-catM", required=False, type=int, help="number of categories")
    parser.add_argument("-p", required=False, type=float, help="poisson scale")
    parser.add_argument("-show", required=False, action='store_true', help="Add this flag if you want to display the heatmap")
    parser.add_argument("-cust", required=False, action='store_true', help="Add this flag for true file containing true labels")
    parser.add_argument("-seed", required=False, type=int, help="set a seed value")

    args = parser.parse_args()


    seed = np.random.randint(1, 2**31 - 1) if args.seed == None else args.seed
    # seed = np.random.randint(1, 2**31 - 1)
    # seed = 82   
    np.random.seed(seed) # should not be same as in learn file

    print("Seed: %s" % seed)

    N = args.n
    D = args.d 
    K_true = args.k 

    mu_scale = 5 if args.m == None else args.m  # bigger the this value more seperated the clusters will be
    covar_scale = 0.1 if args.c == None else args.c 

    z_true = np.random.randint(0, K_true, N) # true assignments
    X = np.zeros((N, D))

    ################################### Generate Gaussian data ###################################

    if D > 0:
        mu = np.zeros((K_true, D))
        sigma = np.zeros((K_true, D, D))

        covar_separation_scale = 0.3 # distance in covariance of a clusters
        covar_separation = covar_separation_scale * np.array([i for i in range(2, K_true+1)])
        covar_separation = np.append(covar_separation, 1)

        # covar_separation = covar_separation_scale ** np.array([i for i in range(1, K_true+1)])

        # Generate random NIW true hyperparameters

        m_true = mu_scale*np.random.randn(K_true, D)

        S_true = np.zeros((K_true, D, D))
        for k in range(K_true):
            # S_true[k] = covar_scale*np.random.rand(D)*np.identity(D)
            S_true[k, :, :] = abs(np.random.normal(covar_scale*covar_separation[k], 0.1, D))*np.identity(D)
            # S_true[k, :, :] = covar_separation[k]*np.eye(D)
            # S_true[k] = np.eye(D)
            # A = np.random.rand(D, D)
            # S_true[k, :, :] = np.dot(A, A.transpose())
        k_true = 0.005 * np.random.randint(1, 30, K_true)
        v_true = np.random.randint(D+1, 2*D + 1, K_true)
        # v_true = (D+3)*np.ones(K_true, int)

        # Sample mu and sigma from NIW distribution
        for k in range(K_true):
            # sigma[k, :, :] = np.random.normal(covar_scale, 0.1, D)*np.identity(D)
            sigma[k, :, :] =  np.diag(invwishart.rvs(df = v_true[k], scale = S_true[k]))*np.identity(D)
            # sigma[k, :, :] =  invwishart.rvs(df = v_true[k], scale = S_true[k])
            if D == 1:     
                mu[k] = np.random.normal(m_true[k], sigma[k]/k_true[k])
            else:
                mu[k] = np.random.multivariate_normal(m_true[k], sigma[k]/k_true[k])

        # Sample data from Multivariate Normal distribution
        for k in range(K_true):
            if D==1:
                X[np.where(z_true == k)] = np.random.normal(mu[k].flatten()[0], sigma[k].flatten()[0]/k_true[k], z_true.tolist().count(k))[:, np.newaxis]
            else:
                X[np.where(z_true == k)] = np.random.multivariate_normal(mu[k], sigma[k]/k_true[k], z_true.tolist().count(k))

    else:
        mu = None
        sigma = None

    ################################# Generate Categorical data #####################################
    cat_dimensions = 0 if args.cat == None else args.cat
    cat_num = 4 if args.catM == None else args.catM
    if cat_dimensions > 0:
        cat_num = 4 if args.catM == None else args.catM
        categorical_Ms = cat_num + np.zeros(cat_dimensions, int)
        for i in range(cat_dimensions):
            # alpha_K = np.array([1, 1])
            alpha_K = 0.01*np.random.randint(20, 60, K_true)
            categorical_M = categorical_Ms[i]

            categorical_mixed_data = np.zeros(N, int)
            for k in range(K_true):
                dirichletSample_prob = np.random.dirichlet(alpha_K[k] + np.zeros(categorical_M), size=1)[0]
                categorical_mixed_data[np.where(z_true == k)] = np.random.choice(categorical_M, p = dirichletSample_prob, size=z_true.tolist().count(k))

            X = np.concatenate([X, [[i] for i in categorical_mixed_data]], axis=1)


    poisson_scale = 0 if args.p == None else args.p
    if poisson_scale > 0:
        lambdas = np.random.randint(10, 100, K_true)
        poisson_data = np.zeros(N, int)
        for k in range(K_true):
            poisson_data[np.where(z_true == k)] = np.random.poisson(lambdas[k], z_true.tolist().count(k))
        
        X = np.concatenate([X, [[i] for i in poisson_data]], axis=1)

    
    


    # covar_scale = np.array([0.2, 100])
    # mu = np.random.randn(D, K_true)*mu_scale # mu for each clusters, (D * K)
    # X = mu[:, z_true] + np.random.randn(D, N)*covar_scale[z_true] # generating a matrix (D * N)
    # X = X.T
    # print(X)

    filename = f"data_n{N}_d{D}_k{K_true}_m{mu_scale}_c{covar_scale}_catD{cat_dimensions}_catM{cat_num}_seed{seed}"

    # ############### Custom mu and sigma #########################
    # mu = np.array([[-1, 4],
    #                [2, 6],
    #                [3, 1],
    #                [2, -3]])

    # sigma = np.array([[[1.72, 0. ],
    #                    [0. , 0.4]],
    #                   [[0.31, 0. ],
    #                    [0. , 2.7]],
    #                   [[1.29, 0. ],
    #                    [0. , 1.2]],
    #                   [[0.19, 0. ],
    #                    [0. , 0.05]]    
    #                   ])

    # # Sample data from Multivariate Normal distribution
    # for k in range(K_true):
    #     X[np.where(z_true == k)] = np.random.multivariate_normal(mu[k], sigma[k], z_true.tolist().count(k))

    # filename = f"data_n{N}_d{D}_k{K_true}_custom"



    ################################## Save data to a file ##################################
    outDir = 'data' if args.o == None else args.o

    if not outDir in os.listdir():
        os.mkdir(outDir)

    with open(f'{outDir}/{filename}.csv', 'w') as f1:
        for data_vec in X:
            data_vec = [str(i) for i in data_vec]
            f1.write(",".join(data_vec)+'\n')

    params_true = {
        "filename":filename,
        "seed": seed,
        "z": z_true,
        "mu": mu,
        "sigma": sigma
    } 

    # with open(f"data/trueParams_{filename}.p", 'wb') as f2:
    #     pickle.dump(params_true, f2, pickle.HIGHEST_PROTOCOL)

    with open(f"{outDir}/{filename}.trueParamPickle", 'wb') as f2:
        pickle.dump(params_true, f2, pickle.HIGHEST_PROTOCOL)
    
    print(f"The created data points are stored in {outDir}/{filename}.csv. The seed used is {seed} (saved in the following pickle file)")
    print(f"The created data parameters is stored in {outDir}/{filename}.trueParamPickle")

    if D == 2:
        # print("m_0: ",*m_true)
        # print("k_0: ",*k_true)
        # print("S_0: ",*S_true)
        # print("V_0: ",*v_true)
        print("mu: ", *mu)
        print("Sigma: ", *sigma)
        print("\n")


    # ################# plotting ############
    # if not 'data_plots' in os.listdir():
    #     os.mkdir('data_plots')

    # print(np.min(X), np.max(X))
    # plt.imshow(X[np.argsort(z_true)])
    # plt.title("True Clustered Data")
    # plt.savefig(f'data_plots/true_heatmap_data_{filename}.png')
    # print(f"The image is saved in: data_plots/true_heatmap_data_{filename}.png")

    # if args.show:
    #     plt.show()
    # plt.close()

    # plt.imshow(X)
    # plt.title("GMM data")
    # plt.savefig(f'data_plots/GMM_heatmap_data_{filename}.png')
    # print(f"The image is saved in: data_plots/GMM_heatmap_data_{filename}.png")

    # if args.show:
    #     plt.show()
    # plt.close()