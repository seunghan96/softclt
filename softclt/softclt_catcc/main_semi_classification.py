import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import *
from models.TC import TC
from models.model import base_Model
from trainer.trainer import *
from utils import _calc_metrics, copy_Files
from utils import _logger, set_requires_grad

start_time = datetime.now()

def densify(x, tau, alpha):
    return ((2*alpha) / (1 + np.exp(-tau*x))) + (1-alpha)*np.eye(x.shape[0])

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--seed',                       default=1,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_linear, gen_pseudo_labels')

parser.add_argument('--selected_dataset',           default='HAR',              type=str,   help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--data_path',                  default=r'data/',           type=str,   help='Path containing dataset')
parser.add_argument('--data_perc',                  default=1,                  type=int,   help='data percentage')
parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='7',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')

############################################################################################################################################
parser.add_argument('--lambda_', default=0.5, type=float)
parser.add_argument('--lambda_aux', type=float, default=0.5)

parser.add_argument('--tau_temp', type=float, default=1)
parser.add_argument('--tau_inst', type=float, default=6)
parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--save_epoch', type=int, default=20)
parser.add_argument('--load_epoch', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=999)

parser.add_argument('--dist_metric', type=str, default='DTW')

############################################################################################################################################
args = parser.parse_args()

device = torch.device(f'cuda:{args.device}')
#experiment_description = f"{args.selected_dataset}_experiment"
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.selected_dataset

args.soft_temporal = (args.tau_temp > 0)
args.soft_instance = (args.tau_inst > 0)
logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

settings = f"TEMP{int(args.soft_temporal)}_INST{int(args.soft_instance)}_"
settings += f'tau_temp{float(args.tau_temp)}_tau_inst{float(args.tau_inst)}_'
settings += f'lambda{args.lambda_}_lambda_aux{args.lambda_aux}'

#base = os.path.join(logs_save_dir, experiment_description, run_description, settings)
base = os.path.join(logs_save_dir, run_description, settings)
if args.training_mode in ['self_supervised','train_linear']:
    experiment_log_dir = os.path.join(base, training_mode + f"_seed_{SEED}")
else:
    experiment_log_dir = os.path.join(base, f'{args.data_perc}p', training_mode  + f"_seed_{SEED}")
    
print('='*50)    
print(experiment_log_dir)
print('='*50)    
os.makedirs(os.path.join(experiment_log_dir,'saved_models'), exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = os.path.join(args.data_path, data_type)
if data_type not in ['HAR','sleepEDF','Epilepsy']:
    train_dl, test_dl = data_generator_wo_val(data_path, configs, training_mode, int(args.data_perc), args.batch_size)
else:
    train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode, int(args.data_perc), args.batch_size)

logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs, args).to(device)
temporal_contr_model = TC(configs, device).to(device)

if training_mode == 'ft' :
    load_from = os.path.join(base, f'{args.data_perc}p',f"ft_linear_seed_{SEED}")
    chkpoint = torch.load(os.path.join(load_from,"saved_models",f"ckp_{args.load_epoch}.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]

    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == 'ft_linear' :
    load_from = os.path.join(base, f"self_supervised_seed_{SEED}")
    chkpoint = torch.load(os.path.join(load_from, "saved_models", f"ckp_{args.load_epoch}.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "gen_pseudo_labels":
    load_from = os.path.join(base, f"{args.data_perc}p", f'ft_seed_{SEED}')
    chkpoint = torch.load(os.path.join(load_from,"saved_models",f"ckp_{args.load_epoch}.pt"), map_location=device)
    #chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)
    gen_pseudo_labels(model, train_dl, device, data_path, args.data_perc)
    sys.exit(0)

if "train_linear" in training_mode:
    if 'SupCon' in training_mode:
        #load_from = os.path.join(base, f"{args.data_perc}p", f"SupCon_seed_{SEED}")
        load_from = os.path.join(base, f"{args.data_perc}p", f"SupCon_seed_{SEED}")
    else:
        #load_from = os.path.join(base, f"self_supervised_seed_{SEED}")
        load_from = os.path.join(base, f"self_supervised_seed_{SEED}")
    chkpoint = torch.load(os.path.join(load_from,"saved_models",f"ckp_{args.load_epoch}.pt"), map_location=device)
    
    
    
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "SupCon":      
    load_from = os.path.join(base, f"{args.data_perc}p", f'ft_seed_{SEED}')
    chkpoint = torch.load(os.path.join(load_from,"saved_models",f"ckp_{args.load_epoch}.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]      
    model.load_state_dict(pretrained_dict)     
    
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=3e-4)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised" or training_mode == "SupCon":  # to do it only once
    #copy_Files(os.path.join(logs_save_dir, experiment_description, run_description, settings), data_type)
    copy_Files(os.path.join(logs_save_dir, run_description, settings), data_type)

if args.soft_instance:
    MAT_PATH = f'data/{data_type}/{args.dist_metric}.npy'
    if os.path.exists(MAT_PATH):
        sim_mat = np.load(MAT_PATH)
    else:
        print(f"Saving {{args.dist_metric}} ...")
        sim_mat = load_sim_matrix(args.selected_dataset)
        dist_mat = 1-sim_mat
        soft_labels = densify(-dist_mat, args.tau_inst, args.alpha)
        del sim_mat, dist_mat
else:
    soft_labels = 0
    

if data_type not in ['HAR','sleepEDF','Epilepsy']:
    Trainer_wo_val(args, soft_labels, model, temporal_contr_model, model_optimizer, 
                   temporal_contr_optimizer, train_dl, test_dl, device,
                   logger, configs, experiment_log_dir, training_mode)
else:
    Trainer(args, soft_labels, model, temporal_contr_model, model_optimizer, 
            temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
            logger, configs, experiment_log_dir, training_mode)
    
if ('linear' in training_mode) | (training_mode == 'ft'):
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, os.path.join(experiment_log_dir), args.home_path, args.num_epochs)

logger.debug(f"Training time is : {datetime.now() - start_time}")
