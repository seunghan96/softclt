import numpy as np
import pandas as pd
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc }

def eval_semi_classification(model, 
                             train_data1, train_labels1, 
                             train_data5, train_labels5,
                             train_labels,
                             test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    
    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    
    if train_labels1.ndim == 2:
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)
        
    train_repr1 = model.encode(train_data1, encoding_window='full_series' if train_labels.ndim == 1 else None)
    train_repr5 = model.encode(train_data5, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    if train_labels.ndim == 2:
        train_repr1 = merge_dim01(train_repr1)
        train_labels1 = merge_dim01(train_labels1)
        train_repr5 = merge_dim01(train_repr5)
        train_labels5 = merge_dim01(train_labels5)

    clf1 = fit_clf(train_repr1, train_labels1)
    clf5 = fit_clf(train_repr5, train_labels5)

    acc1 = clf1.score(test_repr, test_labels)
    acc5 = clf5.score(test_repr, test_labels)
    
    if eval_protocol == 'linear':
        y_score1 = clf1.predict_proba(test_repr)
        y_score5 = clf5.predict_proba(test_repr)
    else:
        y_score1 = clf1.decision_function(test_repr)
        y_score5 = clf5.decision_function(test_repr)
    
    y_pred1 = clf1.predict(test_repr)
    y_pred5 = clf5.predict(test_repr)
    
    if y_score1.ndim==1:
        y_score1 = y_score1.reshape(-1, 1)
    if y_score5.ndim==1:
        y_score5 = y_score5.reshape(-1, 1)
        
    auprc1 = average_precision_score(test_labels_onehot, y_score1)
    auprc5 = average_precision_score(test_labels_onehot, y_score5)
    
    
    f1_1 = f1_score(test_labels, y_pred1, average='macro')
    f1_5 = f1_score(test_labels, y_pred5, average='macro')

    
    return [y_score1,y_score5], { 'acc': [acc1,acc5], 'auprc': [auprc1,auprc5],
                                 'f1': [f1_1,f1_5]}