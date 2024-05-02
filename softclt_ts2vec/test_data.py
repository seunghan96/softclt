import logging
import pickle
import csv
import numpy as np
import pandas as pd
import sys

for mode in ['train' , 'val', 'test']:
    with open(f'/data1/r20user2/EHR_dataset/output/ihm/{mode}_p2x_data.pkl', 'rb') as file:
        data = pickle.load(file)

    print(data[0])
    dim = 17
    a = 0
    ts_buffer = []
    label_buffer = []
    for p_data in data:
        reg_ts = p_data['reg_ts'] # (48,34)
        reg_ts = reg_ts[:,:dim] # (48,17)
        label = p_data['label'] # (1,)
        ts_buffer.append(reg_ts)
        label_buffer.append(label)

    ts_buffer = np.array(ts_buffer)
    label_buffer = np.array(label_buffer)
    ts_filename = f'/home/r20user19/Documents/softclt/softclt_ts2vec/datasets/pheno/{mode}_data.npy'
    label_filename = f'/home/r20user19/Documents/softclt/softclt_ts2vec/datasets/pheno/{mode}_label.csv'
    np.save(ts_filename, ts_buffer)
    np.savetxt(label_filename, label_buffer, delimiter=',')







# loading results for UEA dataset
#  % python train.py Cricket --loader='UEA' --batch-size 8 --gpu 5 --eval 
# Dataset: Cricket
# Arguments: Namespace(alpha=0.5, batch_size=8, dataset='Cricket', dist_type='DTW', epochs=None, eval=True, expid=2, gpu=5, irregular=0, iters=None, lambda_=0.5, loader='UEA', lr=0.001, max_train_length=3000, repr_dims=320, save_every=None, seed=None, separate_reg=False, tau_inst=0, tau_temp=0)
# Loading data... Saving & Loading DTW ...
# Preprocessing MTS ...
# 100%|██████████████████████████████████████████████████████████████| 108/108 [01:08<00:00,  1.57it/s]
# Epoch #0: loss=7.3993108089153585
# Epoch #1: loss=2.87894397515517
# Epoch #2: loss=2.4550544115213246
# Epoch #3: loss=2.2747933864593506
# Epoch #4: loss=2.147034947688763
# Epoch #5: loss=1.9745070017301118
# Epoch #6: loss=1.8259540062684279
# Epoch #7: loss=1.7251459543521588
# Epoch #8: loss=1.6452062405072725
# Epoch #9: loss=1.5006373662215013
# Epoch #10: loss=1.564482597204355
# Epoch #11: loss=1.4997850289711585
# Epoch #12: loss=1.4290230916096613
# Epoch #13: loss=1.2659694644121022
# Epoch #14: loss=1.1573529977064867
# Epoch #15: loss=1.4197673430809608
# Epoch #16: loss=1.1434912039683416
# Epoch #17: loss=1.0234589714270372
# Epoch #18: loss=1.10607065145786
# Epoch #19: loss=1.208749234676361
# Epoch #20: loss=1.0987792015075684
# Epoch #21: loss=1.3050033312577467
# Epoch #22: loss=1.233806706391848
# Epoch #23: loss=1.0182069677572985
# Epoch #24: loss=1.0128918840334966
# Epoch #25: loss=0.9873800598658048
# Epoch #26: loss=0.87138137450585
# Epoch #27: loss=0.8168879884939927
# Epoch #28: loss=1.1612961980012746
# Epoch #29: loss=0.9714335890916678
# Epoch #30: loss=1.126144753052638
# Epoch #31: loss=0.9336813963376559
# Epoch #32: loss=0.9456255252544696
# Epoch #33: loss=0.8479550893490131
# Epoch #34: loss=0.9376987218856812
# Epoch #35: loss=0.860452991265517
# Epoch #36: loss=0.9207309667880719
# Epoch #37: loss=1.03497307575666
# Epoch #38: loss=0.9508903301679171
# Epoch #39: loss=0.8243928735072796
# Epoch #40: loss=0.8784535068732041
# Epoch #41: loss=1.1002224683761597
# Epoch #42: loss=0.9373911435787494
# Epoch #43: loss=0.9389008283615112
# Epoch #44: loss=0.8064643465555631
# Epoch #45: loss=0.7529191466478201

# Training time: 0:02:16.403843

# Evaluation result: {'acc': 0.9861111111111112, 'auprc': 0.998015873015873}
# Finished.
