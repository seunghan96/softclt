import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss, SupConLoss

# (densify)
# 1) model_train
# 2) model_train_wo_DTW
# 3) model_evaluate
# 4) gen_pseudo_labels

def densify(x, tau, alpha=0.5):
    return ((2*alpha) / (1 + np.exp(-tau*x))) + (1-alpha)*np.eye(x.shape[0])

def model_train(soft_labels, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode, lambda_aux):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    soft_labels = torch.tensor(soft_labels, device=device)

    for _, (idx, data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        aug1 = aug1*100
        aug2 = aug2*100
        
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised" or training_mode == "SupCon":
            
            soft_labels_batch = soft_labels[idx][:,idx]

            _, _, features1, features2, final_loss = model(aug1, aug2, soft_labels_batch)
            del soft_labels_batch

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2


        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            Sup_contrastive_criterion = SupConLoss(device)

            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             labels) * lambda2

        else:
            output = model(data, 0, 0, train=False)
            predictions, _ = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        if (training_mode == "self_supervised") :
            loss += lambda_aux*final_loss
        
        if training_mode == 'SupCon':
            loss += 0.1*lambda_aux*final_loss
        
            
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_train_wo_DTW(dist_func, dist_type, tau_inst, model, temporal_contr_model, 
                       model_optimizer, temp_cont_optimizer, criterion, train_loader,
                       config, device, training_mode, lambda_aux):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for _, (_, data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        aug1 = aug1*100
        aug2 = aug2*100
        
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised" or training_mode == "SupCon":
            temp = data.view(data.shape[0], -1).detach().cpu().numpy()
            dist_mat_batch = dist_func(temp)
            if dist_type=='euc':
                dist_mat_batch = (dist_mat_batch - np.min(dist_mat_batch)) / (np.max(dist_mat_batch) - np.min(dist_mat_batch))
                dist_mat_batch = - dist_mat_batch
            dist_mat_batch = densify(dist_mat_batch, tau_inst, alpha=0.5)
            dist_mat_batch = torch.tensor(dist_mat_batch, device=device)
            _, _, features1, features2, final_loss = model(aug1, aug2, dist_mat_batch)
            del dist_mat_batch

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2


        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            Sup_contrastive_criterion = SupConLoss(device)

            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             labels) * lambda2

        else:
            output = model(data, 0, 0, train=False)
            predictions, _ = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        if (training_mode == "self_supervised") :
            loss += lambda_aux*final_loss
        
        if training_mode == 'SupCon':
            loss += 0.1*lambda_aux*final_loss
        
            
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for _, data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data, 0, 0, train=False)

            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, outs, trgs


def gen_pseudo_labels(model, dataloader, device, experiment_log_dir, pc):
    model.eval()
    softmax = nn.Softmax(dim=1)

    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for _, data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            output = model(data, 0, 0, train=False)
            predictions, features = output

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.LongTensor(torch.from_numpy(all_pseudo_labels).long())
    file_name = f"pseudo_train_data_{str(pc)}perc.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")
