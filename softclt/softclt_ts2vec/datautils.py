import os
import numpy as np
import pandas as pd
import torch
from utils import pkl_load, pad_nan_to_target, split_with_nan
from scipy.io.arff import loadarff
from utils_distance_matrix import *
from sklearn.preprocessing import StandardScaler

def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data

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

def normalize_TS_for_varying_length(TS):
    assert TS.ndim == 2
    print('Preprocessing UTS ...')
    TS = [t[~np.isnan(t)] for t in TS]
    TS = [(t-t.min())/(t.max()-t.min()) for t in TS]
    return TS

def normalize_TS_for_varying_length_zero_pad(TS):
    assert TS.ndim == 2
    TS = np.nan_to_num(TS, nan=0.0)
    TS = [(t-t.min())/(t.max()-t.min()) for t in TS]
    return TS


def load_UCR(dataset,type_='DTW'):
    vary_L_dataset = ['AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','GestureMidAirD1',
                      'GestureMidAirD2','GestureMidAirD3','GesturePebbleZ1','GesturePebbleZ2',
                      'PickupGestureWiimoteZ','PLAID','ShakeGestureWiimoteZ', 'DodgerLoopDay', 
                      'DodgerLoopWeekend','DodgerLoopGame']
    vary_L = dataset in vary_L_dataset
    warping = type_ in ['TAM','DTW','FastDTW']

    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)
    
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])
    
    DIST_MAT_PATH = os.path.join('distance_matrix/UCR', dataset)
    os.makedirs(DIST_MAT_PATH, exist_ok=True)    
    DIST_MAT_PATH = os.path.join(DIST_MAT_PATH,f'{type_}.npy')
    
    if os.path.exists(DIST_MAT_PATH):
        print(f"Loading {type_} ...")
        sim_mat = np.load(DIST_MAT_PATH)
    else:
        print(f"Saving & Loading {type_} ...")
        if vary_L:
            if warping:
                norm_TS = normalize_TS_for_varying_length(train)
            else:
                norm_TS = normalize_TS_for_varying_length_zero_pad(train)
        else:
            norm_TS = normalize_TS(train)
        sim_mat = save_sim_mat(norm_TS, min_ = 0, max_ = 1, type_=type_)
        np.save(DIST_MAT_PATH, sim_mat)
    
    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels, sim_mat
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels, sim_mat


def load_UEA(dataset,max_train_length,type_='DTW'):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    DIST_MAT_PATH = os.path.join('distance_matrix/UEA', dataset)
    os.makedirs(DIST_MAT_PATH, exist_ok=True)
    
    DIST_MAT_PATH = os.path.join(DIST_MAT_PATH,f'{type_}.npy')
    
    if os.path.exists(DIST_MAT_PATH):
        print(f"Loading {type_} ...")
        sim_mat = np.load(DIST_MAT_PATH)
    else:
        print(f"Saving & Loading {type_} ...")
        if max_train_length is not None:
            sections = train_X.shape[1] // max_train_length
            if sections >= 2:
                train_X = np.concatenate(split_with_nan(train_X, sections, axis=1), axis=0)
        
        sim_mat = save_sim_mat(normalize_TS(train_X), min_ = 0, max_ = 1, multivariate=True)
        np.save(DIST_MAT_PATH, sim_mat)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y, sim_mat
    

def load_semi_SSL(dataset, type_ = 'DTW'):
    ##################################################################################
    ## LOAD DATASET
    ##################################################################################
    tr = torch.load(f'datasets/SemiSL/{dataset}/train.pt')
    tr_perc1 = torch.load(f'datasets/SemiSL/{dataset}/train_1perc.pt')
    tr_perc5 = torch.load(f'datasets/SemiSL/{dataset}/train_5perc.pt')
    ts = torch.load(f'datasets/SemiSL/{dataset}/test.pt')
    
    # (1) Train (100%)
    train = tr['samples'].detach().cpu().numpy().astype(np.float64)
    train_labels = tr['labels'].detach().cpu().numpy()
    
    # (2) Train (1%)
    train_perc1 = tr_perc1['samples'].detach().cpu().numpy().astype(np.float64)
    train_perc1_labels = tr_perc1['labels'].detach().cpu().numpy()
    
    # (3) Train (5%)
    train_perc5 = tr_perc5['samples'].detach().cpu().numpy().astype(np.float64)
    train_perc5_labels = tr_perc5['labels'].detach().cpu().numpy()
    
    # (4) Test (100%)
    test = ts['samples'].detach().cpu().numpy().astype(np.float64)
    test_labels = ts['labels'].detach().cpu().numpy()
    
    ##################################################################################
    ## ONE-HOT ENCODING
    ##################################################################################
    labels = np.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train_labels = np.vectorize(transform.get)(train_labels)
    train_perc1_labels = np.vectorize(transform.get)(train_perc1_labels)
    train_perc5_labels = np.vectorize(transform.get)(train_perc5_labels)
    test_labels = np.vectorize(transform.get)(test_labels)
    
    ##################################################################################
    ## RESHAPE
    ##################################################################################
    if (train.ndim==3) & (train.shape[1]==1):
        train=train.squeeze(1)
        train_perc1=train_perc1.squeeze(1)
        train_perc5=train_perc5.squeeze(1)
        test=test.squeeze(1)
    elif (train.ndim==3) & (train.shape[1]>1):
        train=train.transpose(0,2,1)
        train_perc1=train_perc1.transpose(0,2,1)
        train_perc5=train_perc5.transpose(0,2,1)
        test=test.transpose(0,2,1)  
    
    DIST_MAT_PATH = os.path.join('distance_matrix/semi_SSL', dataset)
    os.makedirs(DIST_MAT_PATH, exist_ok=True)
    
    DIST_MAT_PATH = os.path.join(DIST_MAT_PATH,f'{type_}.npy')
        
    if os.path.exists(DIST_MAT_PATH):
        print(f"{type_} already exists")
        sim_mat = np.load(DIST_MAT_PATH)
    else:
        print(f"Saving {type_} ...")
        sim_mat = save_sim_mat(normalize_TS(train), min_ = 0, max_ = 1)
        np.save(DIST_MAT_PATH, sim_mat)
    
    ##################################################################################
    ## NORMALIZATION
    ##################################################################################
    if (train.ndim==2):
        if dataset not in [
            'AllGestureWiimoteX',
            'AllGestureWiimoteY',
            'AllGestureWiimoteZ',
            'BME',
            'Chinatown',
            'Crop',
            'EOGHorizontalSignal',
            'EOGVerticalSignal',
            'Fungi',
            'GestureMidAirD1',
            'GestureMidAirD2',
            'GestureMidAirD3',
            'GesturePebbleZ1',
            'GesturePebbleZ2',
            'GunPointAgeSpan',
            'GunPointMaleVersusFemale',
            'GunPointOldVersusYoung',
            'HouseTwenty',
            'InsectEPGRegularTrain',
            'InsectEPGSmallTrain',
            'MelbournePedestrian',
            'PickupGestureWiimoteZ',
            'PigAirwayPressure',
            'PigArtPressure',
            'PigCVP',
            'PLAID',
            'PowerCons',
            'Rock',
            'SemgHandGenderCh2',
            'SemgHandMovementCh2',
            'SemgHandSubjectCh2',
            'ShakeGestureWiimoteZ',
            'SmoothSubspace',
            'UMD'
        ]:
            return train[..., np.newaxis], train_labels, train_perc1[..., np.newaxis], train_perc1_labels, train_perc5[..., np.newaxis], train_perc5_labels, test[..., np.newaxis], test_labels, sim_mat
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels, sim_mat   
    else:
        scaler = StandardScaler()
        scaler.fit(train.reshape(-1, train.shape[-1]))
        train = scaler.transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
        train_perc1 = scaler.transform(train_perc1.reshape(-1, train_perc1.shape[-1])).reshape(train_perc1.shape)
        train_perc5 = scaler.transform(train_perc5.reshape(-1, train_perc5.shape[-1])).reshape(train_perc5.shape)
        test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
        return train, train_labels, train_perc1, train_perc1_labels, train_perc5, train_perc5_labels, test, test_labels, sim_mat
        
        
def load_forecast_npy(name, max_train_length, univar=False, type_='DTW'):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    train_data = data[train_slice]
    
    DIST_MAT_PATH = os.path.join('distance_matrix/forecast', name)
    os.makedirs(DIST_MAT_PATH, exist_ok=True)
        
    DIST_MAT_PATH = os.path.join(DIST_MAT_PATH,f'{type_}.npy')
    
    if os.path.exists(DIST_MAT_PATH):
        print(f"Loading {type_} ...")
        sim_mat = np.load(DIST_MAT_PATH)
    else:
        print(f"Saving & Loading {type_} ...")
        if max_train_length is not None:
            sections = train_data.shape[0] // max_train_length
            if sections >= 2:
                if train_data.ndim==2:
                    temp = np.expand_dims(train_data,0)
                else:
                    temp = train_data
                temp = np.concatenate(split_with_nan(temp, sections, axis=1), axis=0)
            else:
                temp = train_data
        if train_data.ndim==2:
            sim_mat = save_sim_mat(normalize_TS(temp), min_ = 0, max_ = 1)
        else:
            sim_mat = save_sim_mat(normalize_TS(temp), min_ = 0, max_ = 1, multivariate=True)
        np.save(DIST_MAT_PATH, sim_mat)
        
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, sim_mat, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, max_train_length, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    print('data',data.shape)
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name, max_train_length, cold=False):
    res = pkl_load(f'datasets/{name}.pkl')
    if cold:
        train_data = 0
    else:
        train_data = gen_ano_train_data(res['all_train_data'])
        print(train_data.shape)
    return train_data, res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']

def load_semi_SSL(dataset,type_ = 'DTW'):
    tr = torch.load(f'datasets/semi/{dataset}/train.pt')
    tr_perc1 = torch.load(f'datasets/semi/{dataset}/train_1perc.pt')
    tr_perc5 = torch.load(f'datasets/semi/{dataset}/train_5perc.pt')
    ts = torch.load(f'datasets/semi/{dataset}/test.pt')
    
    train = tr['samples'].detach().cpu().numpy().astype(np.float64)
    train_labels = tr['labels'].detach().cpu().numpy()
    train_perc1 = tr_perc1['samples'].detach().cpu().numpy().astype(np.float64)
    train_perc1_labels = tr_perc1['labels'].detach().cpu().numpy()
    train_perc5 = tr_perc5['samples'].detach().cpu().numpy().astype(np.float64)
    train_perc5_labels = tr_perc5['labels'].detach().cpu().numpy()
    
    test = ts['samples'].detach().cpu().numpy().astype(np.float64)
    test_labels = ts['labels'].detach().cpu().numpy()
    
    DIST_MAT_PATH = os.path.join('distance_matrix/semi_SSL', dataset)
    os.makedirs(DIST_MAT_PATH, exist_ok=True)
    
    labels = np.unique(train_labels)
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train_labels = np.vectorize(transform.get)(train_labels)
    train_perc1_labels = np.vectorize(transform.get)(train_perc1_labels)
    train_perc5_labels = np.vectorize(transform.get)(train_perc5_labels)
    test_labels = np.vectorize(transform.get)(test_labels)
    
    
    DIST_MAT_PATH = os.path.join(DIST_MAT_PATH, f'{type_}.npy')
    if (train.ndim==3) & (train.shape[1]==1):
        train=train.squeeze(1)
        train_perc1=train_perc1.squeeze(1)
        train_perc5=train_perc5.squeeze(1)
        test=test.squeeze(1)
    elif (train.ndim==3) & (train.shape[1]>1):
        train=train.transpose(0,2,1)
        train_perc1=train_perc1.transpose(0,2,1)
        train_perc5=train_perc5.transpose(0,2,1)
        test=test.transpose(0,2,1)  
        print(2)
        
        
    if os.path.exists(DIST_MAT_PATH):
        sim_mat = np.load(DIST_MAT_PATH)
    else:
        print(f"Saving {type_} ...")
        sim_mat = save_sim_mat(normalize_TS(train), min_ = 0, max_ = 1)
        np.save(DIST_MAT_PATH, sim_mat)
    
    if (train.ndim==2):
        if dataset not in [
            'AllGestureWiimoteX',
            'AllGestureWiimoteY',
            'AllGestureWiimoteZ',
            'BME',
            'Chinatown',
            'Crop',
            'EOGHorizontalSignal',
            'EOGVerticalSignal',
            'Fungi',
            'GestureMidAirD1',
            'GestureMidAirD2',
            'GestureMidAirD3',
            'GesturePebbleZ1',
            'GesturePebbleZ2',
            'GunPointAgeSpan',
            'GunPointMaleVersusFemale',
            'GunPointOldVersusYoung',
            'HouseTwenty',
            'InsectEPGRegularTrain',
            'InsectEPGSmallTrain',
            'MelbournePedestrian',
            'PickupGestureWiimoteZ',
            'PigAirwayPressure',
            'PigArtPressure',
            'PigCVP',
            'PLAID',
            'PowerCons',
            'Rock',
            'SemgHandGenderCh2',
            'SemgHandMovementCh2',
            'SemgHandSubjectCh2',
            'ShakeGestureWiimoteZ',
            'SmoothSubspace',
            'UMD'
        ]:
            
            return train[..., np.newaxis], train_labels, train_perc1[..., np.newaxis], train_perc1_labels, train_perc5[..., np.newaxis], train_perc5_labels, test[..., np.newaxis], test_labels, sim_mat
            
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels, sim_mat   
    else:
        scaler = StandardScaler()
        scaler.fit(train.reshape(-1, train.shape[-1]))
        
        train = scaler.transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
        train_perc1 = scaler.transform(train_perc1.reshape(-1, train_perc1.shape[-1])).reshape(train_perc1.shape)
        train_perc5 = scaler.transform(train_perc5.reshape(-1, train_perc5.shape[-1])).reshape(train_perc5.shape)
        test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)

        return train, train_labels, train_perc1, train_perc1_labels, train_perc5, train_perc5_labels, test, test_labels, sim_mat
