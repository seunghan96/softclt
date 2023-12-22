import os
import sys

sys.path.append("..")
import torch
import torch.nn as nn

from trainer.train_utils import *
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# 1) Trainer
# 2) Trainer_wo_DTW
# 3) Trainer_wo_val

def Trainer(args, DTW, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    logger.debug("Training started ....")

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    
    if training_mode =='self_supervised':
        num_epochs = args.num_epochs
    else:
        num_epochs = args.load_epoch
    '''        
    if ('linear' in training_mode)|('tl' in training_mode):
        num_epochs = args.num_epochs_linear
    else:
        num_epochs = args.num_epochs
    '''    
    # (2) Train
    os.makedirs(os.path.join(experiment_log_dir,  "saved_models"), exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = model_train(DTW, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, args.lambda_aux)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        if epoch%args.save_epoch==0:
            chkpoint = {'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))
            #if (training_mode != "self_supervised"):
            #    torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.load_epoch}',"saved_models", f'ckp_{epoch}.pt'))
            #else:
            #    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))

    # (3) Save Results
    
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    #if (training_mode != "self_supervised"):
    #    torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.load_epoch}',"saved_models", f'ckp_last.pt'))
    #else:
    #    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", 'ckp_last.pt'))

    # (4) (Optional) Evaluation
    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")

def Trainer_wo_DTW(args, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    logger.debug("Training started ....")

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if training_mode =='self_supervised':
        num_epochs = args.num_epochs
    else:
        num_epochs = args.load_epoch
    '''        
    if ('linear' in training_mode)|('tl' in training_mode):
        num_epochs = args.num_epochs_linear
    else:
        num_epochs = args.num_epochs
    '''    
    ########################################################
    if args.dist == 'cos':
        dist_func = cosine_similarity
    elif args.dist == 'euc':
        dist_func = euclidean_distances
    ########################################################
    
    # (2) Train
    os.makedirs(os.path.join(experiment_log_dir,  "saved_models"), exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = model_train_wo_DTW(dist_func, args.dist, args.tau_inst, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, args.lambda_aux)
        
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        if epoch%args.save_epoch==0:
            chkpoint = {'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            #torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))
            #torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.num_epochs}_load_{args.load_epoch}',"saved_models", f'ckp_{epoch}.pt'))
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))
            #if (training_mode != "self_supervised"):
            #    torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.load_epoch}',"saved_models", f'ckp_{epoch}.pt'))
            #else:
            #    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))

    # (3) Save Results
    #os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    #torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    #torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.num_epochs}_load_{args.load_epoch}',"saved_models", f'ckp_last.pt'))
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    #if (training_mode != "self_supervised"):
    #    torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.load_epoch}',"saved_models", f'ckp_last.pt'))
    #else:
    #    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    # (4) (Optional) Evaluation
    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")


def Trainer_wo_val(args, DTW, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    logger.debug("Training started ....")

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if training_mode =='self_supervised':
        num_epochs = args.num_epochs
    else:
        num_epochs = args.load_epoch
    '''        
    if ('linear' in training_mode)|('tl' in training_mode):
        num_epochs = args.num_epochs_linear
    else:
        num_epochs = args.num_epochs
    '''     
    # (2) Train
    os.makedirs(os.path.join(experiment_log_dir,  "saved_models"), exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = model_train(DTW, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, args.lambda_aux)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}')

        if epoch%args.save_epoch==0:
            chkpoint = {'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            #torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))
            #torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.num_epochs}_load_{args.load_epoch}',"saved_models", f'ckp_{epoch}.pt'))
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))
            #if (training_mode != "self_supervised"):
            #    torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.load_epoch}',"saved_models", f'ckp_{epoch}.pt'))
            #else:
            #    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))

    # (3) Save Results
    #os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    #torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    #torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.num_epochs}_load_{args.load_epoch}',"saved_models", f'ckp_last.pt'))
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    #if (training_mode != "self_supervised"):
    #    torch.save(chkpoint, os.path.join(experiment_log_dir, f'ep_pretrain_{args.load_epoch}',"saved_models", f'ckp_last.pt'))
    #else:
    #    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    # (4) (Optional) Evaluation
    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")
    

def Trainer_wo_DTW_wo_val(args, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    logger.debug("Training started ....")

    # (1) Loss Function & LR Scheduler & Epochs
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if training_mode =='self_supervised':
        num_epochs = args.num_epochs
    else:
        num_epochs = args.num_epochs_finetune

    ########################################################
    if args.dist == 'cos':
        dist_func = cosine_similarity
    elif args.dist == 'euc':
        dist_func = euclidean_distances
    ########################################################
    
    # (2) Train
    os.makedirs(os.path.join(experiment_log_dir,  "saved_models"), exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = model_train_wo_DTW(dist_func, args.dist, args.tau_inst, model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode, args.lambda_aux)
        
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}')

        if epoch%args.save_epoch==0:
            chkpoint = {'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_{epoch}.pt'))


    # (3) Save Results
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))


    # (4) (Optional) Evaluation
    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")
