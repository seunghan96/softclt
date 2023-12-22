import os
import tqdm

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from tslearn.metrics import dtw, dtw_path,gak

import torch
from torch.utils.data import Dataset

from .augmentations import DataTransform
from sklearn.preprocessing import MinMaxScaler

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
    N = dist_mat.shape[0]
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


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return index, self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return index, self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode, pc, batch_size):
    #batch_size = configs.batch_size

    if training_mode != "SupCon":
        if ('ft' in training_mode) & (pc == 1):
            print('1%')
            train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"))
        elif ('ft' in training_mode) & (pc == 5):
            print('5%')
            train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
        elif ('ft' in training_mode) & (pc == 10):
            print('10%')
            train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
        elif ('ft' in training_mode) & (pc == 50):
            print('50%')
            train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
        elif ('ft' in training_mode) & (pc == 75):
            print('75%')
            train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
        else:
            train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    else :
        train_dataset = torch.load(os.path.join(data_path, f"pseudo_train_data_{str(pc)}perc.pt"))
    
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    print('Train size: ',train_dataset['samples'].shape)
    print('Valid size: ',valid_dataset['samples'].shape)
    print('Test size: ',test_dataset['samples'].shape)
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    if batch_size == 999:
        if train_dataset.__len__() < batch_size:
            if train_dataset.__len__() > 16:
                batch_size = 16
            else:
                batch_size = 4
        else:
            batch_size = configs.batch_size

    print('batch_size',batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    return train_loader, valid_loader, test_loader


def data_generator_wo_val(data_path, configs, training_mode, pc, batch_size):
    #batch_size = configs.batch_size

    if training_mode != "SupCon":
        if ('ft' in training_mode) & (pc == 1):
            print('1%')
            train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"))
        elif ('ft' in training_mode) & (pc == 5):
            print('5%')
            train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
        elif ('ft' in training_mode) & (pc == 10):
            print('10%')
            train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
        elif ('ft' in training_mode) & (pc == 50):
            print('50%')
            train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
        elif ('ft' in training_mode) & (pc == 75):
            print('75%')
            train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
        else:
            train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    else :
        train_dataset = torch.load(os.path.join(data_path, f"pseudo_train_data_{str(pc)}perc.pt"))
    
#    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    train_dataset['samples'] = train_dataset['samples']#[:1000,:,:]
    test_dataset['samples'] = test_dataset['samples']#[:1000,:,:]
    train_dataset['labels'] = train_dataset['labels']#[:1000]
    test_dataset['labels'] = test_dataset['labels']#[:1000]
    print('Train size: ',train_dataset['samples'].shape)
    print('Test size: ',test_dataset['samples'].shape)
    
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    if train_dataset.__len__() < configs.batch_size:
        if train_dataset.__len__() > 16:
            batch_size = 16
        else:
            batch_size = 4
    else:
        batch_size = configs.batch_size
            
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    return train_loader,test_loader

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def normalize_TS(TS):
    TS = set_nan_to_zero(TS)
    if TS.ndim == 2:
        print('Preprocessing UTS ...')
        TS_max = TS.max(axis = 1).reshape(-1,1)
        TS_min = TS.min(axis = 1).reshape(-1,1)
        TS = (TS - TS_min)/(TS_max - TS_min + (1e-6))        
    elif TS.ndim == 3:
        print('Preprocessing MTS ...')
        N, D, L = TS.shape
        TS_max = TS.max(axis=2).reshape(N,D,1) 
        TS_min = TS.min(axis=2).reshape(N,D,1)
        TS = (TS - TS_min) / (TS_max - TS_min + (1e-6))   
    return TS


def load_sim_matrix(dataset, type_='DTW'):
    tr = torch.load(f'data/{dataset}/train.pt')
    train = tr['samples'].detach().cpu().numpy().astype(np.float64)
    
    if (train.ndim==3) & (train.shape[1]==1):
        train=train.squeeze(1)
    elif (train.ndim==3) & (train.shape[1]>1):
        train=train.transpose(0,2,1)
    
    os.makedirs(f'data/{dataset}', exist_ok=True)
    MAT_PATH = os.path.join(f'data/{dataset}',f'{type_}.npy')
        
    if os.path.exists(MAT_PATH):
        print(f"{type_} already exists")
        sim_mat = np.load(MAT_PATH)
    else:
        print(f"Saving {type_} ...")
        sim_mat = save_sim_mat(normalize_TS(train), min_ = 0, max_ = 1)
        np.save(MAT_PATH, sim_mat)
        
    return sim_mat