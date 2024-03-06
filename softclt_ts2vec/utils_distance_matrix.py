import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
from tslearn.metrics import dtw, dtw_path,gak
import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

def tam(path, report='full'):
    # Delay and advance counting
    delay = len(find(np.diff(path[0]) == 0))
    advance = len(find(np.diff(path[1]) == 0))

    # Phase counting
    incumbent = find((np.diff(path[0]) == 1) * (np.diff(path[1]) == 1))
    phase = len(incumbent)

    # Estimated and reference time series duration.
    len_estimation = path[1][-1]
    len_ref = path[0][-1]

    p_advance = advance * 1. / len_ref
    p_delay = delay * 1. / len_estimation
    p_phase = phase * 1. / np.min([len_ref, len_estimation])

    return p_advance + p_delay + (1 - p_phase)

def get_DTW(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = dtw(UTS_tr[i].reshape(-1,1), UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_TAM(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                k = dtw_path(UTS_tr[i].reshape(-1,1), 
                             UTS_tr[j].reshape(-1,1))[0]
                p = [np.array([i[0] for i in k]),
                     np.array([i[1] for i in k])]
                dist = tam(p)
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_GAK(UTS_tr):
    N = len(UTS_tr)
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                dist = gak(UTS_tr[i].reshape(-1,1), 
                           UTS_tr[j].reshape(-1,1))
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat


def get_MDTW(MTS_tr):
    N = MTS_tr.shape[0]
    dist_mat = np.zeros((N,N))
    for i in tqdm.tqdm(range(N)):
        for j in range(N):
            if i>j:
                mdtw_dist = dtw(MTS_tr[i], MTS_tr[j])
                dist_mat[i,j] = mdtw_dist
                dist_mat[j,i] = mdtw_dist
            elif i==j:
                dist_mat[i,j] = 0
            else :
                pass
    return dist_mat

def get_COS(MTS_tr):
    cos_sim_matrix = -cosine_similarity(MTS_tr)
    return cos_sim_matrix

def get_EUC(MTS_tr):
    return euclidean_distances(MTS_tr)

def save_sim_mat(X_tr, min_ = 0, max_ = 1, multivariate=False, type_='DTW'):
    if multivariate:
        assert type=='DTW'
        dist_mat = get_MDTW(X_tr)
    else:
        if type_=='DTW':
            dist_mat = get_DTW(X_tr)
        elif type_=='TAM':
            dist_mat = get_TAM(X_tr)
        elif type_=='COS':
            dist_mat = get_COS(X_tr)
        elif type_=='EUC':
            dist_mat = get_EUC(X_tr)
        elif type_=='GAK':
            dist_mat = get_GAK(X_tr)
    N = dist_mat.shape[0]
        
    # (1) distance matrix
    diag_indices = np.diag_indices(N)
    mask = np.ones(dist_mat.shape, dtype=bool)
    mask[diag_indices] = False
    temp = dist_mat[mask].reshape(N, N-1)
    dist_mat[diag_indices] = temp.min()
    
    # (2) normalized distance matrix
    scaler = MinMaxScaler(feature_range=(min_, max_))
    dist_mat = scaler.fit_transform(dist_mat)
    
    # (3) normalized similarity matrix
    return 1 - dist_mat 

def get_example_data(data_name):
    ex_data_path = f'./data/UCR/{data_name}/{data_name}_TRAIN.tsv'
    data = pd.read_csv(ex_data_path, delimiter='\t', keep_default_na=False, header=None)
    data_X = data.iloc[:,1:]
    data_y = data.iloc[:,0]
    return data_X,data_y
    
def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def densify(x, tau, alpha):
    return ((2*alpha) / (1 + np.exp(-tau*x))) + (1-alpha)*np.eye(x.shape[0])

def topK_one_else_zero(matrix, k):
    """
    Convert the off-diagonal elements to one if they are in the top-k values of each row, and zero if not
    """
    new_matrix = matrix.copy()
    top_k_indices = np.argpartition(new_matrix, -(k+1), axis=1)[:, -(k+1):]
    np.fill_diagonal(new_matrix, 0)
    
    mask = np.zeros_like(new_matrix)
    mask[np.repeat(np.arange(new_matrix.shape[0]), (k+1)),
         top_k_indices.flatten()] = 1
    return mask

def convert_hard_matrix(soft_matrix, pos_ratio):
    N = soft_matrix.shape[0]
    num_pos = int((N-1) * pos_ratio)
    hard_matrix = topK_one_else_zero(soft_matrix, num_pos)
    return hard_matrix