import pickle
from sklearn.metrics.cluster import adjusted_rand_score
import argparse
import matplotlib.pyplot as plt
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t", required=True, type=argparse.FileType('rb'), help="Pickle file name containing true model parameters")
parser.add_argument("-p", required=True, type=argparse.FileType('rb'), help="Pickle file name containing predicted model parameters")
parser.add_argument("-cust", required=False, type=str, help="Add this flag for true file containing true labels")

args = parser.parse_args()

params_true_z = pickle.load(args.t)['z']

if args.cust == "p":
    params_pred_z = []
    for line in args.p:
        params_pred_z.append(line.strip())    
else:
    params_pred = pickle.load(args.p)
# cont_table = utils.contingency_table(params_true_z, params_pred['z'])
# print("Contigency table:")
# print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in cont_table]))

# import json
# with open("../results/Z_true.json", 'w') as f:
#     json.dump(params_true['z'].tolist(), f)

# mod, file  = args.p.name.split('/')
# file = os.path.splitext(file)[0]
# print(f"{mod},{file},{len(set(params_pred['z']))},{adjusted_rand_score(params_pred['z'], params_true['z'])},{params_pred['time']}")

if args.cust:
    print(f"K:{len(set(params_pred_z))},ARI:{round(adjusted_rand_score(params_pred_z, params_true_z),4)}")

else:
    print(f"Real Time Taken: {params_pred['time']}")
    print(f"Predicted K: {len(set(params_pred['z']))}")
    print(f"ARI score with map Z: {round(adjusted_rand_score(params_pred['z'], params_true_z), 4)}")
#    print(f"ARI score with last iter Z: {round(adjusted_rand_score(params_pred['z_last_iter'], params_true_z), 4)}")
