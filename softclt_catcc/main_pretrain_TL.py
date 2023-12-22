import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import os
from datetime import datetime

import numpy as np
import torch

from dataloader.dataloader import *
from models.TC import TC
from models.model import base_Model
from trainer.trainer import *
from utils import _logger

start_time = datetime.now()

######################## Model parameters ########################
parser = argparse.ArgumentParser()
home_dir = os.getcwd()
parser.add_argument('--seed',                       default=1,                  type=int,   help='seed value')
parser.add_argument('--selected_dataset',           default='SleepEEG',              type=str)
parser.add_argument('--data_path',                  default=r'data/',           type=str,   help='Path containing dataset')
parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='7',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')

parser.add_argument('--lambda_', default=0.5, type=float)
parser.add_argument('--lambda_aux', default=0.5, type=float)
parser.add_argument('--alpha', type=float, default=0.5)

parser.add_argument('--tau_temp', type=float, default=1)
parser.add_argument('--tau_inst', type=float, default=1)

parser.add_argument('--num_epochs', type=int, default=40)
parser.add_argument('--save_epoch', type=int, default=10)

parser.add_argument('--dist', type=str, default='cos')
############################################################################################################################################

args = parser.parse_args()
assert args.selected_dataset == 'SleepEEG'
args.soft_instance = (args.tau_inst > 0)
args.soft_temporal = (args.tau_temp > 0)

device = torch.device(f'cuda:{args.device}')
data_type = args.selected_dataset
training_mode = 'self_supervised'

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

settings = f"INST{int(args.soft_instance)}_TEMP{int(args.soft_temporal)}_"
settings += f"inst{args.tau_inst}_temp{args.tau_temp}_"
settings += f"lambda{args.lambda_}_lambda_aux{args.lambda_aux}_{args.dist}"

log_dir = os.path.join(logs_save_dir, args.selected_dataset, 
                                  settings,  training_mode + f"_seed_{SEED}")
os.makedirs(os.path.join(log_dir,'saved_models'), exist_ok=True)
print(log_dir)

counter = 0
src_counter = 0

log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
data_path = os.path.join(args.data_path, data_type)
label_pct = 100
train_dl, test_dl = data_generator_wo_val(data_path, configs, training_mode, 
                                          label_pct, configs.batch_size)

logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs, args).to(device)
temporal_contr_model = TC(configs, device).to(device)
    
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                   weight_decay=3e-4)

temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    
Trainer_wo_DTW_wo_val(args, model, temporal_contr_model, model_optimizer, 
                      temporal_contr_optimizer, train_dl, test_dl, device,
                      logger, configs, log_dir, training_mode)

logger.debug(f"Training time is : {datetime.now() - start_time}")

