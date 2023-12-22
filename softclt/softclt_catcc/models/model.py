import torch
from torch import nn

from models.timelags import *
from models.soft_losses import *
from models.hard_losses import *

class base_Model(nn.Module):
    def __init__(self, configs, args):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
        self.lambda_ = args.lambda_ # 0.5
        self.soft_temporal = args.soft_temporal #True
        self.soft_instance = args.soft_instance #True
        
        self.tau_temp = args.tau_temp
        

    def forward(self, aug1, aug2, soft_labels, train=True):
        if train:
            if self.soft_instance:
                soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
                del soft_labels
            
            temporal_loss = torch.tensor(0., device=aug1.device)
            instance_loss = torch.tensor(0., device=aug1.device)
            
            #-------------------------------------------------#
            # DEPTH = 1
            #-------------------------------------------------#
            aug1 = self.conv_block1(aug1)
            aug2 = self.conv_block1(aug2)
            if self.soft_temporal:
                d=0
                timelag = timelag_sigmoid(aug1.shape[2], self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=aug1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
            else:
                temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
            if self.soft_instance:
                instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
            else:
                instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)

            #-------------------------------------------------#
            # DEPTH = 2
            #-------------------------------------------------#
            aug1 = self.conv_block2(aug1)
            aug2 = self.conv_block2(aug2)

            if self.soft_temporal:
                d=1
                timelag = timelag_sigmoid(aug1.shape[2],self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=aug1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
            else:
                temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
                
            if self.soft_instance:
                instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
            else:
                instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)
            
            #-------------------------------------------------#
            # DEPTH = 3
            #-------------------------------------------------#
            aug1 = self.conv_block3(aug1)
            aug2 = self.conv_block3(aug2)
        
            if self.soft_temporal:
                d=2
                timelag = timelag_sigmoid(aug1.shape[2],self.tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=aug1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                temporal_loss += (1-self.lambda_) * temp_CL_soft(aug1, aug2, timelag_L, timelag_R)
            else:
                temporal_loss += (1-self.lambda_) * temp_CL_hard(aug1, aug2)
           
            if self.soft_instance:
                instance_loss += self.lambda_ * inst_CL_soft(aug1, aug2, soft_labels_L, soft_labels_R)
                del soft_labels_L, soft_labels_R
            else:
                instance_loss += self.lambda_ * inst_CL_hard(aug1, aug2)
        
        else:
            aug = self.conv_block1(aug1)
            aug = self.conv_block2(aug)
            aug = self.conv_block3(aug)
        
        ############################################################################
        ############################################################################
        
        if train:
            aug1_flat = aug1.reshape(aug1.shape[0], -1)
            aug2_flat = aug2.reshape(aug2.shape[0], -1)
            aug1_logits = self.logits(aug1_flat)
            aug2_logits = self.logits(aug2_flat)
            final_loss = temporal_loss + instance_loss
            return aug1_logits, aug2_logits, aug1, aug2, final_loss

        else:
            aug_flat = aug.reshape(aug.shape[0], -1)
            aug_logits = self.logits(aug_flat)
            return aug_logits, aug
                