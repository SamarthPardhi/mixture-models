import pickle
from sklearn.metrics.cluster import adjusted_rand_score
import argparse
import matplotlib.pyplot as plt
import utils
import numpy as np
import os
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, type=argparse.FileType('r'), help="Dataset file name")
parser.add_argument("-t", required=False, type=argparse.FileType('rb'), help="Pickle file name containing true model parameters")
parser.add_argument("-p", required=False, type=argparse.FileType('rb'), help="Pickle file name containing predicted model parameters")
parser.add_argument("-o", required=False, type=str, help="Output directory")
parser.add_argument("-show", required=False, action='store_true', help="Add this flag if you want to display the plots")
parser.add_argument("-cust", required=False, action='store_true', help="Add this flag for file containing only labels")

args = parser.parse_args()

dataFile = args.f
dataParamFile = args.t
modelParamFile = args.p

if dataFile != None:
        
    X = []

    dataFilename = os.path.splitext(os.path.basename(args.f.name))[0]
    print(dataFilename)

    for line in args.f:
        X.append(np.array([float(i) for i in line.strip().split(',')]))
    X = np.array(X)

    N = len(X)
    D = len(X[0])
    outDir = "plotsBin" if args.o == None else args.o

    if outDir not in os.listdir():
        os.mkdir(outDir)
            
    ################################ GMM Data ###################################
    if D>=0:
        fig, ax = plt.subplots(figsize=(10, 20))
        heatmap = sns.heatmap(X)
        ax.set_title(f"{dataFilename}")
        ax.set_xlabel('Features')
        ax.set_ylabel('Samples')
        
        plt.savefig(f'{outDir}/{dataFilename}.png')
        print(f"The image is saved in: {outDir}/{dataFilename}.png")

        if args.show:
            plt.show()
        plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # utils.plot_mixture_model(ax, X, pickle.load(dataParamFile)['z'])

        # plt.savefig(f'{outDir}/{dataFilename}.png')
        # print(f"The image is saved in: {outDir}/{dataFilename}.png")

        # if args.show:
        #     plt.show()
        # plt.close()

    ############################### True Clustered Data ##################################

    if dataParamFile != None:

        if D>=0:
            fig, ax = plt.subplots(figsize=(10, 20))

            if not args.cust:
                params_true = pickle.load(args.t)['z']
            else:
                params_true = []
                for line in args.t:
                    params_true.append(line.strip())

            heatmap = sns.heatmap(X[np.argsort(params_true)])
            ax.set_title(f"true_{dataFilename}")
            ax.set_xlabel('features')
            ax.set_ylabel('samples')

            plt.savefig(f'{outDir}/true_{dataFilename}.png')
            print(f'The image is saved in: {outDir}/true_{dataFilename}.png')

            if args.show:
                plt.show()
            plt.close()

    if modelParamFile != None:
        
        params_pred_dic = pickle.load(args.p)
        modelFilename = os.path.splitext(os.path.basename(args.p.name))[0]
        
        ############################### Predicted Clustered Data ##################################

        if D > 0:
            params_pred = params_pred_dic['z']
            K = len(set(params_pred))
            modelTime = params_pred_dic['time']
            try:
                ARI_score = round(adjusted_rand_score(params_pred, params_true), 2)
            except:
                ARI_score = 'None'

            fig, ax = plt.subplots(figsize=(10, 20))
            heatmap = sns.heatmap(X[np.argsort(params_pred)])
            ax.set_title(f"K_pred:{K}  N:{N} ARI_Score:{ARI_score} Time:{round(modelTime, 2)}s    {modelFilename}")
            ax.set_xlabel('features')
            ax.set_ylabel('samples')
            
            plt.savefig(f'{outDir}/model_{modelFilename}.png')
            print(f"The image is saved in: {outDir}/model_{modelFilename}.png")
        
        # else:
        #     if D == 2:
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111)

        #         utils.plot_mixture_model(ax, X, params_true['z'])
        #         K = len(set(params_pred['z']))
        #         plt.title(f"K_pred:{K}  N:{N}   ARI_Score:{round(adjusted_rand_score(params_pred['z'], params_true['z']), 2)}")
        #         if len(params_pred['mu']) > 0:   
        #             for k in range(K):
        #                 utils.plot_ellipse(ax, params_pred['mu'][k], params_pred['sigma'][k])
        #         plt.savefig(f'outputs_img/2D_fit_{modelFilename}.png')
        #         print(f"The image is saved in: outputs_img/2D_fit_{modelFilename}.png")

        #     elif D == 1:
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111)

        #         K_true = len(set(params_true['z']))
        #         utils.plot_hist_mm(ax, X, K_true, params_pred['z'])
        #         plt.show()
        #         plt.close()
        #     else:
        #         pass

        if args.show:
            plt.show()
        plt.close()
