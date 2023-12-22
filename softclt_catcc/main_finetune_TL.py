import argparse
import os
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import *
from models.TC import TC
from models.model import base_Model
from trainer.trainer import *
from utils import _calc_metrics
from utils import _logger, set_requires_grad

start_time = datetime.now()


######################## Model parameters ########################
parser = argparse.ArgumentParser()
home_dir = os.getcwd()
parser.add_argument('--seed',                       default=1,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_linear, gen_pseudo_labels')

parser.add_argument('--source_dataset',           default='SleepEEG',              type=str,   help='Datadset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--target_dataset',           default='HAR',              type=str,   help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--data_path',                  default=r'data/',           type=str,   help='Path containing dataset')
parser.add_argument('--data_perc',                  default=1,                  type=int,   help='data percentage')
parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='7',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')

############################################################################################################################################
parser.add_argument('--lambda_', default=0.5, type=float)
parser.add_argument('--lambda_aux', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('--tau_temp', type=float, default=1)
parser.add_argument('--tau_inst', type=float, default=1)

parser.add_argument('--load_epoch', type=int, default=40)
parser.add_argument('--save_epoch', type=int, default=20)
parser.add_argument('--num_epochs_finetune', type=int, default=20)

parser.add_argument('--batch_size', type=int, default=999)
parser.add_argument('--dist', type=str, default='cos')
############################################################################################################################################

args = parser.parse_args()
args.soft_instance = (args.tau_inst > 0)
args.soft_temporal = (args.tau_temp > 0)
source = args.source_dataset
target = args.target_dataset

device = torch.device(f'cuda:{args.device}')
assert args.training_mode in ['linear_probing','fine_tune']

data_path = os.path.join(args.data_path, target)
training_mode = args.training_mode

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{source}_Configs import Config as Configs')
exec(f'from config_files.{target}_Configs import Config as Configs_target')
configs = Configs()
configs_target = Configs_target()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

settings = f"INST{int(args.soft_instance)}_TEMP{int(args.soft_temporal)}_"
settings += f"inst{args.tau_inst}_temp{args.tau_temp}_"
settings += f"lambda{args.lambda_}_lambda_aux{args.lambda_aux}_{args.dist}"

base = os.path.join(logs_save_dir, f'{source}2{target}', settings)
log_dir = os.path.join(base, training_mode + f"_seed_{SEED}")

os.makedirs(os.path.join(log_dir, 'saved_models'), exist_ok=True)    
print('='*50)    
print(log_dir)
print('='*50)    

counter = 0
src_counter = 0

log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'PRETRAIN Dataset: {source}')
logger.debug(f'FINETUNE Dataset: {target}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)


# Load datasets
train_dl, test_dl = data_generator_wo_val(data_path, configs_target, 
                                          training_mode, int(args.data_perc), args.batch_size)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs_target, args).to(device)
temporal_contr_model = TC(configs_target, device).to(device)

pretrained_weight_path = os.path.join(args.logs_save_dir, args.source_dataset, settings,
                            f"self_supervised_seed_{SEED}","saved_models", f"ckp_{args.load_epoch}.pt")
pre_W = torch.load(pretrained_weight_path, map_location=device)["model_state_dict"]
model_W = model.state_dict()

for k in model_W.keys():
    if (model_W[k].shape != pre_W[k].shape) & ('logit' not in k):
        _, C_target, _ = model_W[k].shape # 3
        _, C_source, _ = pre_W[k].shape # 1
        dup = C_target // C_source
        pre_W[k] = pre_W[k].repeat(1,dup,1)

pre_W_copy = pre_W.copy()
for i in pre_W_copy.keys():
    if 'logits' in i:
        del pre_W[i]

model_W.update(pre_W)
model.load_state_dict(model_W)

if training_mode == "linear_probing" :
    set_requires_grad(model, pre_W, requires_grad=False)  # Freeze everything except last layer.

model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, 
                                   betas=(configs.beta1, configs.beta2),
                                   weight_decay=3e-4)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), 
                                            lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), 
                                            weight_decay=3e-4)

Trainer_wo_DTW_wo_val(args, model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, test_dl, device,
        logger, configs, log_dir, training_mode)

outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
total_loss, total_acc, pred_labels, true_labels = outs
save_setting = f'load_{args.load_epoch}_ft_{args.num_epochs_finetune}_{args.training_mode}'
_calc_metrics(pred_labels, true_labels, 
              os.path.join(log_dir,save_setting), args.home_path)

logger.debug(f"Training time is : {datetime.now() - start_time}")



