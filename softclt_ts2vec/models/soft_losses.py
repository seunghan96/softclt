import torch
import torch.nn.functional as F
from models.timelags import *

########################################################################################################
## 1. Soft Contrastive Losses
########################################################################################################
#------------------------------------------------------------------------------------------#
# (1) Instance-wise CL
#------------------------------------------------------------------------------------------#
def inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    loss = torch.sum(logits[:,i]*soft_labels_L)
    loss += torch.sum(logits[:,B + i]*soft_labels_R)
    loss /= (2*B*T)
    return loss

#------------------------------------------------------------------------------------------#
# (2) Temporal CL
#------------------------------------------------------------------------------------------#
def temp_CL_soft(z1, z2, timelag_L, timelag_R):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    loss = torch.sum(logits[:,t]*timelag_L)
    loss += torch.sum(logits[:,T + t]*timelag_R)
    loss /= (2*B*T)
    return loss

#------------------------------------------------------------------------------------------#
# (3) Hierarchical CL = Instance CL + Temporal CL
# (The below differs by the way it generates timelag for temporal CL )
## 3-1) hier_CL_soft : sigmoid
## 3-2) hier_CL_soft_window : window
## 3-3) hier_CL_soft_thres : threshold
## 3-4) hier_CL_soft_gaussian : gaussian
## 3-5) hier_CL_soft_interval : same interval
## 3-6) hier_CL_soft_wo_inst : 3-1) w/o instance CL
#------------------------------------------------------------------------------------------#

def hier_CL_soft(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, 
                 soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    
    if soft_labels is not None:
        soft_labels = torch.tensor(soft_labels, device=z1.device)
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    if temporal_hierarchy:
                        timelag = timelag_sigmoid(z1.shape[1],tau_temp*(2**d))
                    else:
                        timelag = timelag_sigmoid(z1.shape[1],tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_window(z1, z2, soft_labels, window_ratio, tau_temp=2, lambda_=0.5,
                        temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_sigmoid_window(z1.shape[1],tau_temp*(2**d),window_ratio)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d

def hier_CL_soft_thres(z1, z2, soft_labels, threshold, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_sigmoid_threshold(z1.shape[1], threshold)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_gaussian(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False, temporal_hierarchy=True):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    if temporal_hierarchy:
                        timelag = timelag_gaussian(z1.shape[1],tau_temp/(2**d))
                    else:
                        timelag = timelag_gaussian(z1.shape[1],tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d


def hier_CL_soft_interval(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_same_interval(z1.shape[1],tau_temp/(2**d))
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    return loss / d

def hier_CL_soft_wo_inst(z1, z2, soft_labels, tau_temp=2, lambda_=0.5, temporal_unit=0, soft_temporal=False, soft_instance=False):
    soft_labels = torch.tensor(soft_labels, device=z1.device)
    soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    timelag = timelag_sigmoid(z1.shape[1],tau_temp*(2**d))
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        d += 1

    return loss / d

