
import numpy as np
import time
from . import _eval_protocols as eval_protocols

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
        print('train_repr',train_repr.shape)
        print('train_data',train_data.shape)
        D = train_data.shape[2]
        for d in range(D):            
            print(train_data[0,:,d])
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        t = time.time()
        print('train_features',train_features.shape)
        print('train_labels',train_labels.shape)
        print('valid_features',valid_features.shape)
        print('valid_labels',valid_labels.shape)
        fadsfsd
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
