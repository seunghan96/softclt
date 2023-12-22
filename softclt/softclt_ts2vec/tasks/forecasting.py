import numpy as np
import time
from . import _eval_protocols as eval_protocols

def generate_pred_samples(features, data, pred_len, drop=0):
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])
            
def generate_pred_samples_norm(features, data, pred_len, drop=0):
    n = data.shape[1]

    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i : 1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    last_val = data[:,:labels.shape[1],:] #last_val = data[0,:labels.shape[1],0]
    features = features[:, drop:]
    labels = labels[:, drop:]
    last_val = last_val[:,drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3]), last_val[0,:,:]            

def cal_metrics(pred, target):
    return {
        'MSE': ((pred - target) ** 2).mean(),
        'MAE': np.abs(pred - target).mean()
    }
    
def eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
        
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        
        
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)

        #fadfsd
        t = time.time()

        lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = temp_2d.T
                test_pred_inv = temp_2d.reshape(test_pred.shape) 

                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = temp_2d.T
                test_labels_inv = temp_2d.reshape(test_labels.shape)

            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1)
                temp_2d = scaler.inverse_transform(temp_2d)
                test_pred_inv = temp_2d.reshape(test_pred.shape)
                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                temp_2d = scaler.inverse_transform(temp_2d)
                test_labels_inv = temp_2d.reshape(test_labels.shape)
            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                
            #test_pred_inv = scaler.inverse_transform(test_pred)
            #test_labels_inv = scaler.inverse_transform(test_labels)
        
        out_log[pred_len] = {
            'norm': test_pred,
            'raw': test_pred_inv,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred, test_labels),
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res

def eval_forecasting_norm(model, data, data_not_scaled, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data_not_scaled[:, train_slice, n_covariate_cols:]
    valid_data = data_not_scaled[:, valid_slice, n_covariate_cols:]
    test_data = data_not_scaled[:, test_slice, n_covariate_cols:]

    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:

        train_features, train_labels, train_last_val = generate_pred_samples_norm(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels, valid_last_val = generate_pred_samples_norm(valid_repr, valid_data, pred_len)
        test_features, test_labels, test_last_val = generate_pred_samples_norm(test_repr, test_data, pred_len)
        t = time.time()

        lr = eval_protocols.fit_ridge_norm(train_features, train_labels, train_last_val, valid_features, valid_labels, valid_last_val)
        lr_train_time[pred_len] = time.time() - t
        t = time.time()
        test_pred = lr.predict(test_features)
        try:
            test_pred = test_pred+test_last_val
        except:
            test_pred = test_pred+np.repeat(test_last_val,pred_len,axis=1)
        
        lr_infer_time[pred_len] = time.time() - t
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)

        if test_data.shape[0] > 1:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                
                ######## temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_pred_inv = temp_2d.reshape(test_pred.shape) 

                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                
                ######## temp_2d = scaler.inverse_transform(temp_2d.T)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_labels_inv = temp_2d.reshape(test_labels.shape)

            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                
                ######## test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = scaler.transform(test_pred_reshaped.T)
                test_pred_reshaped = test_pred_reshaped.T
                
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                
                ######## test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = scaler.transform(test_labels_reshaped.T)
                test_labels_reshaped = test_labels_reshaped.T
                
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
        else:
            if univar:
                temp_2d = test_pred.reshape(test_pred.shape[0],-1)

                ######## temp_2d = scaler.inverse_transform(temp_2d)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_pred_inv = temp_2d.reshape(test_pred.shape)
                temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                
                ######## temp_2d = scaler.inverse_transform(temp_2d)
                temp_2d = scaler.transform(temp_2d.T)
                temp_2d = temp_2d.T
                
                test_labels_inv = temp_2d.reshape(test_labels.shape)
            else:
                a,b,c,d = test_pred.shape
                test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                
                ######## test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_pred_reshaped = scaler.transform(test_pred_reshaped)
                
                test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                
                a,b,c,d = test_labels.shape
                test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                
                ######## test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                test_labels_reshaped = scaler.transform(test_labels_reshaped)
                
                test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                
            #test_pred_inv = scaler.inverse_transform(test_pred)
            #test_labels_inv = scaler.inverse_transform(test_labels)
        
        out_log[pred_len] = {
            'norm': test_pred_inv,
            'raw': test_pred,
            'norm_gt': test_labels,
            'raw_gt': test_labels_inv
        }
        ours_result[pred_len] = {
            'norm': cal_metrics(test_pred_inv, test_labels_inv),
            'raw': cal_metrics(test_pred, test_labels)
        }
        
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res


def eval_forecasting2(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
        
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        test_pred_list = []
        test_pred_inv_list = []
        test_labels_list = []
        test_labels_inv_list = []
        norm_metric_list = []
        raw_metric_list = []
        
        D = train_data.shape[2]

        
        for d in range(D):            

            train_features, train_labels = generate_pred_samples(train_repr, np.expand_dims(train_data[:,:,d],-1), pred_len, drop=padding)
            valid_features, valid_labels = generate_pred_samples(valid_repr, np.expand_dims(valid_data[:,:,d],-1), pred_len)
            test_features, test_labels = generate_pred_samples(test_repr, np.expand_dims(test_data[:,:,d],-1), pred_len)
            t = time.time()
            
            lr = eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
            lr_train_time[pred_len] = time.time() - t
            t = time.time()
            test_pred = lr.predict(test_features)
            lr_infer_time[pred_len] = time.time() - t
            ori_shape = test_data.shape[0], -1, pred_len, 1
   
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)

            if test_data.shape[0] > 1:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_pred_inv = temp_2d.reshape(test_pred.shape) 

                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_labels_inv = temp_2d.reshape(test_labels.shape)

                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                    #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
            else:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1)
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    test_pred_inv = temp_2d.reshape(test_pred.shape)
                    
                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    #temp_2d = scaler.inverse_transform(temp_2d)
                    test_labels_inv = temp_2d.reshape(test_labels.shape)
                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    
                #test_pred_inv = scaler.inverse_transform(test_pred)
                #test_labels_inv = scaler.inverse_transform(test_labels)
            
            test_pred_list.append(test_pred)
            test_pred_inv_list.append(test_pred_inv)
            test_labels_list.append(test_labels)
            test_labels_inv_list.append(test_labels_inv)
            
            norm_metric_list.append(cal_metrics(test_pred, test_labels))
            raw_metric_list.append(cal_metrics(test_pred_inv, test_labels_inv))
            
        out_log[pred_len] = {
            'norm': test_pred_list,
            'raw': test_pred_inv_list,
            'norm_gt': test_labels_list,
            'raw_gt': test_labels_inv_list
        }
        
        ours_result[pred_len] = {
            'norm': norm_metric_list,
            'raw': raw_metric_list
        }

    
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    return None , eval_res

def eval_forecasting2_norm(model, data, data_not_scaled, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar):
    padding = 200
    
    t = time.time()
    
    all_repr = model.encode(
        data,
        casual=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=256
    )
    ts2vec_infer_time = time.time() - t
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    train_data = data_not_scaled[:, train_slice, n_covariate_cols:]
    valid_data = data_not_scaled[:, valid_slice, n_covariate_cols:]
    test_data = data_not_scaled[:, test_slice, n_covariate_cols:]
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    for pred_len in pred_lens:
        test_pred_list = []
        test_pred_inv_list = []
        test_labels_list = []
        test_labels_inv_list = []
        norm_metric_list = []
        raw_metric_list = []
        
        D = train_data.shape[2]
 
        
        for d in range(D):            

            train_features, train_labels, train_last_val = generate_pred_samples_norm(train_repr, np.expand_dims(train_data[:,:,d],-1), pred_len, drop=padding)
            valid_features, valid_labels, valid_last_val = generate_pred_samples_norm(valid_repr, np.expand_dims(valid_data[:,:,d],-1), pred_len)
            test_features, test_labels, test_last_val = generate_pred_samples_norm(test_repr, np.expand_dims(test_data[:,:,d],-1), pred_len)
        

            t = time.time()
            lr = eval_protocols.fit_ridge_norm(train_features, train_labels, train_last_val, valid_features, valid_labels, valid_last_val)
            
            lr_train_time[pred_len] = time.time() - t
            t = time.time()
            test_pred = lr.predict(test_features)
            test_pred = test_pred+test_last_val
            lr_infer_time[pred_len] = time.time() - t
            ori_shape = test_data.shape[0], -1, pred_len, 1
  
            test_pred = test_pred.reshape(ori_shape)
            test_labels = test_labels.reshape(ori_shape)

            if test_data.shape[0] > 1:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1) 
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_pred_inv = temp_2d.reshape(test_pred.shape) 

                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    temp_2d = temp_2d.T
                    test_labels_inv = temp_2d.reshape(test_labels.shape)

                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    
                    ########test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = scaler.transform(test_pred_reshaped)

                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    
                    ########test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = scaler.transform(test_labels_reshaped)
                    
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    #test_pred_inv = scaler.inverse_transform(test_pred.swapaxes(0, 3)).swapaxes(0, 3)
                    #test_labels_inv = scaler.inverse_transform(test_labels.swapaxes(0, 3)).swapaxes(0, 3)
            else:
                if univar:
                    temp_2d = test_pred.reshape(test_pred.shape[0],-1)
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d.T)
                    test_pred_inv = temp_2d.reshape(test_pred.shape)
                    
                    temp_2d = test_labels.reshape(test_labels.shape[0],-1)
                    
                    ########temp_2d = temp_2d.T*np.sqrt(scaler.var_[d])+scaler.mean_[d]
                    temp_2d = (temp_2d.T-scaler.mean_[d])/np.sqrt(scaler.var_[d])
                    
                    #temp_2d = scaler.inverse_transform(temp_2d)
                    test_labels_inv = temp_2d.reshape(test_labels.shape)
                else:
                    a,b,c,d = test_pred.shape
                    test_pred_reshaped = np.transpose(test_pred, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_pred_reshaped = np.reshape(test_pred_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_pred_reshaped = test_pred_reshaped.reshape((a*b*c, d))
                    
                    ########test_pred_reshaped = scaler.inverse_transform(test_pred_reshaped)
                    test_pred_reshaped = scaler.transform(test_pred_reshaped)
                    
                    test_pred_reshaped = test_pred_reshaped.reshape((a, b*c, d))
                    test_pred_inv = test_pred_reshaped.reshape(a,b,c,d)
                    
                    a,b,c,d = test_labels.shape
                    test_labels_reshaped = np.transpose(test_labels, (0, 2, 1, 3))  # Transpose the array to shape (a, c, b, d)
                    test_labels_reshaped = np.reshape(test_labels_reshaped, (a, b*c, d))  # Reshape the array to shape (a, b*c, d)
                    test_labels_reshaped = test_labels_reshaped.reshape((a*b*c, d))
                    
                    ########test_labels_reshaped = scaler.inverse_transform(test_labels_reshaped)
                    test_labels_reshaped = scaler.transform(test_labels_reshaped)
                    
                    test_labels_reshaped = test_labels_reshaped.reshape((a, b*c, d))
                    test_labels_inv = test_labels_reshaped.reshape(a,b,c,d)
                    
                #test_pred_inv = scaler.inverse_transform(test_pred)
                #test_labels_inv = scaler.inverse_transform(test_labels)
            
            test_pred_list.append(test_pred)
            test_pred_inv_list.append(test_pred_inv)
            test_labels_list.append(test_labels)
            test_labels_inv_list.append(test_labels_inv)
            
            norm_metric_list.append(cal_metrics(test_pred, test_labels))
            raw_metric_list.append(cal_metrics(test_pred_inv, test_labels_inv))
            
        out_log[pred_len] = {
            'norm': test_pred_inv_list,
            'raw': test_pred_list,
            'norm_gt': test_labels_inv_list,
            'raw_gt': test_labels_list
        }
        
        ours_result[pred_len] = {
            'norm': raw_metric_list,
            'raw': norm_metric_list
        }

    
    eval_res = {
        'ours': ours_result,
        'ts2vec_infer_time': ts2vec_infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    return None , eval_res
