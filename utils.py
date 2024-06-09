from unicodedata import name
from matplotlib.patches import Ellipse
import numpy as np
# import torch.nn.functional as F
# import torch
from scipy.special import logsumexp

colors = np.array([x for x in "rgcmykbgrbgcmykbgrcmykbgrcmyk"])
colors = np.hstack([colors] * 200) # 5600 (change this for huge number of cluster > 560)

# normal-inverse-Wishart distribution object
class NIchi2(object):
    def __init__(self, m_0, k_0, v_0, S_0):
        self.m_0 = m_0
        self.k_0 = k_0
        D = len(m_0)
        # assert v_0 >= D, "v_0 must be larger or equal to dimension of data"
        self.v_0 = v_0
        self.S_0 = S_0

def sample_v2(p_k, k_uni):

    # Samples new k from it's discrete probability dist
    for i in range(len(p_k)):
        if k_uni <  p_k[i]:
            return i
    return len(p_k) - 1

def sample_numpy_gumbel(log_p_k):
    return np.argmax(log_p_k + np.random.gumbel(0, 1, len(log_p_k)))

# def sample_gumbel(log_p_k):
#     prob_z = F.gumbel_softmax(torch.tensor(log_p_k)).numpy()
#     return np.random.choice(len(prob_z), p=prob_z)

def sample(log_p_k):
    prob_z = np.exp(log_p_k - logsumexp(log_p_k))
    return np.random.choice(len(prob_z), p=prob_z)

    # return np.argmax(prob_z)
    k_uni = np.random.random()
    # Samples new k from it's discrete probability dist
    for i in range(len(p_k)):
        if k_uni <  p_k[i]:
            return i
    
    return len(p_k) - 1 

# copy pasted this function
def plot_ellipse(ax, mu, sigma, color="b"):

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)


def plot_mixture_model(ax, X, z_true):
    ax.scatter(X[:, 0], X[:, 1], color=colors[z_true].tolist(), s=10)

def plot_hist_mm(ax, X, K, z_true):
    for k in range(K):
        data = []
        ass =  z_true
        for i in range(len(ass)):
            if ass[i]==k:
                data.append(X[:,0][i])
        ax.hist(data, color=colors[k], bins = np.linspace(min(data), max(data), len(data) - int(np.sqrt(len(data)))), density=1)

        # ax.hist(data, color=colors[k], bins = np.linspace(min(data), max(data), ))

    # ax.hist(X[:,0], color=colors[model.clusters.assignments].tolist(), s=10)

def contingency_table(true_clusters, predicted_clusters):

    num_true_clusters = len(np.unique(true_clusters))
    num_predicted_clusters = len(np.unique(predicted_clusters))
    
    table = np.zeros((num_true_clusters+1, num_predicted_clusters+1), dtype=int)
    
    for i in range(len(true_clusters)):
        true_cluster = true_clusters[i]
        predicted_cluster = predicted_clusters[i]
        table[true_cluster+1, predicted_cluster+1] += 1
        table[true_cluster+1, 0] += 1
        table[0, predicted_cluster+1] += 1
    
    maxargs = np.zeros(len(table), dtype=int)
    for i in range(1, len(table)):
        maxargs[i] = np.argmax(table[i][1:]) + 1

    table = table.T
    if num_true_clusters == num_predicted_clusters:
        table_new = np.zeros((num_true_clusters+1, num_predicted_clusters+1), dtype=int).T

        for i in range(len(table_new)):
            table_new[i,:] = table[maxargs[i]]

        return table_new
    else:
        return table


def generate_separable_clusters(separability):
    # Ensure separability is within a valid range
    if separability <= 0 or separability >= 1:
        return "Separability must be in the range (0, 1)"

    # Define the values and their probabilities for each cluster
    cluster1 = np.random.choice([0, 1], size=100, p=[separability, 1 - separability])
    cluster2 = np.random.choice([1, 2], size=100, p=[separability, 1 - separability])
    cluster3 = np.random.choice([2, 3], size=100, p=[separability, 1 - separability])

    # Concatenate the clusters to create a one-dimensional array
    categorical_data = np.concatenate([cluster1, cluster2, cluster3])

    # # Shuffle the data
    # np.random.shuffle(categorical_data)

    return categorical_data



def saveData(filename, data, remark):

    f = open(filename, 'w')
    if remark.lower() == "data":
        for data_vec in data:
            data_vec = [str(i) for i in data_vec]
            f.write(",".join(data_vec)+'\n')
    elif remark.lower() == "labels":
        for z_i in data:
            f.write(f"{z_i}\n")
    elif remark.lower() == "single":
    	f.write(f"{data}")
    else:
        return "[Error] No proper remark found"

    f.flush()
    return filename        



def extractData(filename, remark):
    
    f = open(filename, "r")
    if remark.lower() == "data":
        X = []
        for line in f:
            X.append(np.array([float(i) for i in line.strip().split(',')]))
        X = np.array(X)
        return X
    
    if remark.lower() == "data_int":
        X = []
        for line in f:
            X.append(np.array([int(i) for i in line.strip().split(',')]))
        X = np.array(X)
        return X
    
    elif remark.lower() == "labels":
        return [int(line.strip()) for line in open(filename, "r")]

    elif remark.lower() == "single":
    	with open(filename, "r") as f:
    	   return f.readline().strip()

    return "[Error] No proper remark found"

