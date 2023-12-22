import numpy as np
import torch

def dup_matrix(mat):
    mat0 = torch.tril(mat, diagonal=-1)[:, :-1]   
    mat0 += torch.triu(mat, diagonal=1)[:, 1:]
    mat1 = torch.cat([mat0,mat],dim=1)
    mat2 = torch.cat([mat,mat0],dim=1)
    return mat1, mat2

##############################################################################
## 6 Different ways of generating time lags
##############################################################################
def timelag_sigmoid(T,sigma=1):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)  # set very small values to 0         
    return matrix

def timelag_gaussian(T,sigma):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = np.exp(-(dist**2)/(2 * sigma ** 2))
    matrix = np.where(matrix < 1e-6, 0, matrix) 
    return matrix

def timelag_same_interval(T):
    d = np.arange(T)
    X, Y = np.meshgrid(d, d)
    matrix = 1 - np.abs(X - Y) / T
    return matrix

def timelag_sigmoid_window(T, sigma=1, window_ratio=1.0):
    dist = np.arange(T)
    dist = np.abs(dist - dist[:, np.newaxis])
    matrix = 2 / (1 +np.exp(dist*sigma))
    matrix = np.where(matrix < 1e-6, 0, matrix)          
    dist_from_diag = np.abs(np.subtract.outer(np.arange(dist.shape[0]), np.arange(dist.shape[1])))
    matrix[dist_from_diag > T*window_ratio] = 0
    return matrix

def timelag_sigmoid_threshold(T, threshold=1.0):
    dist = np.ones((T,T))
    dist_from_diag = np.abs(np.subtract.outer(np.arange(dist.shape[0]), np.arange(dist.shape[1])))
    dist[dist_from_diag > T*threshold] = 0
    return dist

##############################################################################

