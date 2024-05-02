# Soft Contrastive Learning for Time Series

### Seunghan Lee, Taeyoung Park, Kibok Lee

<br>

This repository contains the official implementation for the paper [Soft Contrastive Learning for Time Series]([https://arxiv.org/abs/xxxxx](https://arxiv.org/abs/2312.16424)) 

This work is accepted in 

- [ICLR 2024](https://openreview.net/forum?id=pAsQSWlDUf)
- [NeurIPS 2023 Workshop: Self-Supervised Learning - Theory and Practice](https://sslneurips23.github.io/).

<br>

# 1. SoftCLT + TS2Vec

```bash
chdir softclt_ts2vec
```

Please refer to https://github.com/yuezhihan/ts2vec for

- (1) Requirements
- (2) Dataset preparation

<br>

## Code Example

### 1) Standard Classification

**UCR 128 datasets** for univariate TS classification

```python
bs = 8
data = 'BeetleFly'
# tau_inst = xx
# tau_temp = xx

!python train.py {data} --loader='UCR' --batch-size {bs} --eval \
    --tau_inst {tau_inst} --tau_temp {tau_temp} 
```

<br>

**UEA 30 datasets** for multivariate TS classification

```python
bs = 8
data = 'Cricket'
# tau_inst = xx
# tau_temp = xx

!python train.py {data} --loader='UEA' --batch-size {bs} --eval \
    --tau_inst {tau_inst} --tau_temp {tau_temp} 
```

<br>

For optimal hyperparameter setting for each dataset, please refer to `hyperparameters/cls_hyperparams.csv`

<br>

## 2) Semi-supervised Classification

```python
data = 'Epilepsy'
# bs = xx
# tau_inst = xx
# tau_temp = xx

!python train.py {data} --loader='semi' --batch-size {bs} --eval \
    --tau_inst {tau_inst} --tau_temp {tau_temp}
```

<br>

For optimal hyperparameter setting for each dataset, please refer to

- `hyperparameters/semi_cls_1p_hyperparams.csv`
- `hyperparameters/semi_cls_5p_hyperparams.csv`

<br>

## 3) Anomaly Detection

( Note that we ***only use temporal CL*** for anomaly detection task )

```python
data = 'yahoo'
# bs = xxx
# tau_temp = xxx

!python train.py {data} --loader='anomaly' --batch-size {bs} --eval \
	--lambda_=0 --tau_temp={tau_temp}
```

<br>

For optimal hyperparameter setting for each dataset, please refer to

- `hyperparameters/ad_hyperparams.csv`

<br>

# 2. SoftCLT + TS-TCC/CA-TCC

```bash
chdir softclt_catcc
```

Please refer to https://github.com/emadeldeen24/TS-TCC and https://github.com/emadeldeen24/CA-TCC for

- (1) Requirements
- (2) Dataset preparation

<br>

## 1) Semi-supervised Classification

```python
dataset = 'Epilepsy'
# tau_inst = xxx
# tau_temp = xxx
# lambda_ = xxx
# lambda_aux = xxx

#############################################################
# TS-TCC : (1)~(2)
# CA-TCC : (1)~(7)
#############################################################
# (1) Pretrain
!python main_semi_classification.py --selected_dataset {dataset} --training_mode "self_supervised" \
		--tau_temp {tau_temp} --tau_inst {tau_inst} \
		--lambda_ {lambda_} --lambda_aux {lambda_aux
                    
# (2) Finetune Classifier 
!python3 main_semi_classification.py --selected_dataset {dataset} --training_mode "train_linear" \
    --tau_temp {tau_temp} --tau_inst {tau_inst} \
    --lambda_ {lambda_} --lambda_aux {lambda_aux} 
                        
# (3) Finetune Classifier ( with partially labeled datasets )
# (4) Finetune Encoder ( with partially labeled datasets )
# (5) Generate Pseudo-labels
# (6) Supervised CL
# (7) Finetune Classifier
labeled_pc = 1

for mode_ in ['ft_linear','ft','gen_pseudo_labels','SupCon','train_linear_SupCon']:
  !python3 main_semi_classification.py --selected_dataset {dataset} --training_mode {mode_} \
      --tau_temp {tau_temp} --tau_inst {tau_inst} \
      --lambda_ {lambda_} --lambda_aux {lambda_aux} \
      --data_perc {labeled_pc}     
```

<br>

## 2) Transfer Learning

- Source dataset: SleepEEG
- Target dataset: Epilepsy, FD-B, Gesture, EMG

```python
source_data = 'SleepEEG'
target_data = 'Epilepsy'
epoch_pretrain = 40
# tau_inst = xx
# tau_temp = xx
# lambda = xx
# lambda_aux = xx

!python3 main_pretrain_TL.py --selected_dataset {source_data} \
    --tau_temp {tau_temp} --tau_inst {tau_inst} \
    --num_epochs {epoch_pretrain} \
    --lambda_ {lambda_} --lambda_aux {lambda_aux}
```

<br>

```python
tm = 'fine_tune' # 'linear_probing'
finetune_epoch = 50 # 100,200,300,400

!python3 main_finetune_TL.py --training_mode {tm} \
      --source_dataset {source_data} --target_dataset {target_data} \
      --tau_temp {tau_temp} --tau_inst {tau_inst} \
      --load_epoch {load_epoch} \
      --num_epochs_finetune {ft_epoch}\
      --lambda_ {lambda_} --lambda_aux {lambda_aux}
```

<br>

# Contact

If you have any questions, please contact **seunghan9613@yonsei.ac.kr**

<br>

# Acknowledgement

We appreciate the following github repositories for their valuable code base & datasets:

- https://github.com/yuezhihan/ts2vec
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/emadeldeen24/CA-TCC
