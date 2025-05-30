import argparse
import pickle
import time
import sys
import os

import torch
from proc_utils import Dataset, split_validation
from model import Attention_SessionGraph, train_test # Only import what's needed directly
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    return v.lower() in ('true')

class Diginetica_arg():
    dataset = 'diginetica'
    batchSize = 50
    hiddenSize = 100
    epoch = 30
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = True
    validation = True
    valid_portion = 0.1
    ssl_weight = 0.1
    ssl_temperature = 0.07
    ssl_item_drop_prob = 0.2
    ssl_projection_dim = 50 # hiddenSize // 2
    n_gpu = 0
    max_len = 50
    position_emb_dim = 100

class Yoochoose_arg():
    dataset = 'yoochoose1_64'
    batchSize = 75
    hiddenSize = 120
    epoch = 30
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = True
    validation = True
    valid_portion = 0.1
    ssl_weight = 0.1
    ssl_temperature = 0.07
    ssl_item_drop_prob = 0.2
    ssl_projection_dim = 60 # hiddenSize // 2
    n_gpu = 0
    max_len = 50
    position_emb_dim = 120


def main(opt):
    model_save_dir = 'saved_ssl_time/'
    log_dir = 'logs_ssl_time/'

    for_makedirs = [model_save_dir, log_dir]
    for directory in for_makedirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")

    writer = SummaryWriter(log_dir=log_dir)

    if torch.cuda.is_available():
        n_gpu_available = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpu_available}")
        if opt.n_gpu == 0 and n_gpu_available >= 1:
            print(f"Auto-detected {n_gpu_available} GPUs. Using all available.")
            opt.n_gpu = n_gpu_available
        elif opt.n_gpu > n_gpu_available:
            print(f"Warning: Requested {opt.n_gpu} GPUs, but only {n_gpu_available} are available. Using {n_gpu_available}.")
            opt.n_gpu = n_gpu_available
        device = torch.device("cuda:0" if opt.n_gpu > 0 else "cpu")
        print(f"Using {opt.n_gpu if opt.n_gpu > 0 else 'CPU'}{' GPU(s)' if opt.n_gpu > 0 else ''}. Main device: {device}")
    else:
        device = torch.device("cpu"); opt.n_gpu = 0
        print("CUDA not available. Using CPU.")

    if opt.dataset == 'diginetica':
        train_data_path = 'datasets/cikm16/raw/train.txt'; test_data_path = 'datasets/cikm16/raw/test.txt'
    elif opt.dataset == 'yoochoose1_64':
        train_data_path = 'datasets/yoochoose1_64/raw/train.txt'; test_data_path = 'datasets/yoochoose1_64/raw/test.txt'
    else: print(f"Error: Unknown dataset {opt.dataset}"); sys.exit(1)
    try:
        train_data_raw = pickle.load(open(train_data_path, 'rb'))
        test_data_raw = pickle.load(open(test_data_path, 'rb'))
    except FileNotFoundError: print(f"Error: Dataset file not found at {train_data_path} or {test_data_path}"); sys.exit(1)

    if opt.validation:
        train_data_raw, valid_data_raw = split_validation(train_data_raw, opt.valid_portion)
        test_data_raw = valid_data_raw
        print(f'Using validation set ({len(valid_data_raw[0]) if valid_data_raw and len(valid_data_raw) > 0 else 0} sessions) for testing.')
    else:
        print(f'Using full test set ({len(test_data_raw[0]) if test_data_raw and len(test_data_raw) > 0 else 0} sessions).')

    if not train_data_raw or not train_data_raw[0]:
        print("Error: Training data is empty or not loaded correctly.")
        sys.exit(1)
    print(f'Training set size: {len(train_data_raw[0])} sessions.')


    train_data_loader = Dataset(train_data_raw, shuffle=True, opt=opt)
    test_data_loader = Dataset(test_data_raw, shuffle=False, opt=opt)
    
    actual_dataset_max_len = train_data_loader.len_max
    if opt.max_len < actual_dataset_max_len or opt.max_len == 0 : # If 0, it means auto-detect from dataset default
        print(f"Updating opt.max_len from {opt.max_len} to actual dataset max session length: {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    if opt.position_emb_dim == 0 or opt.position_emb_dim != opt.hiddenSize: # If 0, means use hiddenSize
        if opt.position_emb_dim !=0 and opt.position_emb_dim != opt.hiddenSize:
             print(f"Adjusting opt.position_emb_dim from {opt.position_emb_dim} to opt.hiddenSize ({opt.hiddenSize}) for additive combination.")
        elif opt.position_emb_dim == 0:
             print(f"Setting opt.position_emb_dim to opt.hiddenSize ({opt.hiddenSize}) as it was 0.")
        opt.position_emb_dim = opt.hiddenSize


    if opt.dataset == 'diginetica': n_node = 43098
    elif opt.dataset == 'yoochoose1_64': n_node = 37484
    else: n_node = 310; print(f"Warning: n_node using fallback {n_node}")

    model_instance = Attention_SessionGraph(opt, n_node)

    if opt.n_gpu > 1 and torch.cuda.is_available():
        print(f"Wrapping model with torch.nn.DataParallel for {opt.n_gpu} GPUs.")
        model = torch.nn.DataParallel(model_instance, device_ids=list(range(opt.n_gpu)))
        model.to(device)
    else: model = model_instance.to(device)

    print(f"Model initialized. n_node={n_node}, max_session_len_for_pos_emb={opt.max_len}, pos_emb_dim={opt.position_emb_dim}")
    print(f"Hyperparameters: {vars(opt)}")

    start_time = time.time()
    best_result = [0.0, 0.0]; best_epoch = [0, 0]; bad_counter = 0

    for epoch_num in range(opt.epoch):
        print('-' * 50 + f'\nEpoch: {epoch_num}')
        current_hit, current_mrr = train_test(model, train_data_loader, test_data_loader, opt, device)
        print(f'Epoch {epoch_num} Eval - Recall@20: {current_hit:.4f}, MRR@20: {current_mrr:.4f}')
        writer.add_scalar('epoch/recall_at_20', current_hit, epoch_num)
        writer.add_scalar('epoch/mrr_at_20', current_mrr, epoch_num)
        
        flag = 0
        saved_this_epoch_path = ""

        if current_hit > best_result[0] - 1e-5 :
            if abs(current_hit - best_result[0]) > 1e-5 or current_mrr > best_result[1] - 1e-5:
                best_result[0] = current_hit; best_epoch[0] = epoch_num; flag = 1
                saved_this_epoch_path = model_save_dir + f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt'
                state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_to_save, saved_this_epoch_path)
                print(f"Saved best model (by Recall or improving MRR) to {saved_this_epoch_path}")

        if current_mrr > best_result[1] - 1e-5 :
             if abs(current_mrr - best_result[1]) > 1e-5 or current_hit > best_result[0] - 1e-5 :
                if not (abs(current_hit - best_result[0]) < 1e-5 and abs(current_mrr - best_result[1]) < 1e-5): # Avoid re-setting if already best by recall condition
                    best_result[1] = current_mrr; best_epoch[1] = epoch_num; flag = 1
                
                mrr_save_path = model_save_dir + f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt'
                if mrr_save_path != saved_this_epoch_path : # Avoid saving twice if path is identical
                    state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                    torch.save(state_to_save, mrr_save_path)
                    print(f"Saved best model (by MRR or improving Recall) to {mrr_save_path}")
                elif not saved_this_epoch_path and flag == 1 : # if it became the best for MRR and wasn't saved due to recall
                    state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                    torch.save(state_to_save, mrr_save_path)
                    print(f"Saved best model (by MRR) to {mrr_save_path}")

        print(f'Current Best: Recall@20: {best_result[0]:.4f} (Epoch {best_epoch[0]}), MRR@20: {best_result[1]:.4f} (Epoch {best_epoch[1]})')
        bad_counter += (1 if flag == 0 else 0)
        if bad_counter >= opt.patience:
            print(f"Early stopping after {opt.patience} epochs without improvement."); break
            
    writer.close()
    print('-' * 50 + f"\nTotal Running time: {(time.time() - start_time):.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ds_group = parser.add_argument_group('Dataset Configuration')
    ds_group.add_argument('--dataset', default='diginetica', choices=['diginetica', 'yoochoose1_64'], help='Dataset name')
    ds_group.add_argument('--validation', type=str2bool, default=None, help='Use validation split (Default: True for dataset classes)')
    ds_group.add_argument('--valid_portion', type=float, default=-1.0, help='Validation split portion (Default: 0.1 for dataset classes)')

    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--hiddenSize', type=int, default=0, help='Hidden state dimension (0: use dataset class default)')
    model_group.add_argument('--step', type=int, default=0, help='GNN propagation steps (0: use dataset class default)')
    model_group.add_argument('--nonhybrid', type=str2bool, default=None, help='Use non-hybrid scoring (Default: True for dataset classes)')
    model_group.add_argument('--max_len', type=int, default=0, help='Max session length for pos_emb (0: use dataset class default or auto-detect)')
    model_group.add_argument('--position_emb_dim', type=int, default=0, help='Dimension for position embeddings (0: use hiddenSize)')

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--defaults', type=str2bool, default=True, help='Use all default configurations from the chosen dataset class. CMD args can override.')
    train_group.add_argument('--n_gpu', type=int, default=0, help='Num GPUs (0: auto/CPU, >0: specific count)')
    train_group.add_argument('--batchSize', type=int, default=0, help='Batch size (0: use dataset class default)')
    train_group.add_argument('--epoch', type=int, default=0, help='Number of epochs (0: use dataset class default)')
    train_group.add_argument('--lr', type=float, default=0.0, help='Learning rate (0.0: use dataset class default)')
    train_group.add_argument('--lr_dc', type=float, default=0.0, help='LR decay rate (0.0: use dataset class default)')
    train_group.add_argument('--lr_dc_step', type=int, default=0, help='Steps for LR decay (0: use dataset class default)')
    train_group.add_argument('--l2', type=float, default=-1.0, help='L2 penalty (-1.0: use dataset class default, to distinguish from 0.0)')
    train_group.add_argument('--patience', type=int, default=0, help='Early stopping patience (0: use dataset class default)')

    ssl_group = parser.add_argument_group('Self-Supervised Learning (SSL) Hyperparameters')
    ssl_group.add_argument('--ssl_weight', type=float, default=-1.0, help='SSL loss weight (-1.0: use dataset class default)')
    ssl_group.add_argument('--ssl_temperature', type=float, default=-1.0, help='SSL InfoNCE temperature (-1.0: use dataset class default)')
    ssl_group.add_argument('--ssl_item_drop_prob', type=float, default=-1.0, help='SSL item drop probability (-1.0: use dataset class default)')
    ssl_group.add_argument('--ssl_projection_dim', type=int, default=-1, help='SSL projection head dim (-1: use dataset class default, 0 means hiddenSize/2)')

    cmd_opt = parser.parse_args()
    
    # Determine the base configuration class
    if cmd_opt.dataset == 'diginetica':
        base_config_class = Diginetica_arg
    elif cmd_opt.dataset == 'yoochoose1_64':
        base_config_class = Yoochoose_arg
    else:
        print(f"FATAL: Unknown dataset '{cmd_opt.dataset}' for loading base configurations.")
        sys.exit(1)
    
    base_config = base_config_class()
    
    # Initialize final 'opt' with attributes from base_config_class
    # This ensures all necessary attributes are present.
    opt = argparse.Namespace(**vars(base_config))

    # If --defaults is True, explicit command-line arguments override the class defaults.
    # If --defaults is False, command-line arguments are primary;
    #   if a CMD arg is not provided (left at its parser default), then the class default is used.
    
    if cmd_opt.defaults:
        print("Using dataset default configurations. Explicit command-line arguments will override specific defaults.")
        for key, cmd_value in vars(cmd_opt).items():
            # Override if the cmd_value is different from its parser's default,
            # ensuring user-provided values take precedence.
            if cmd_value != parser.get_default(key):
                if hasattr(opt, key):
                    # print(f"Overriding default '{key}' from '{getattr(opt, key)}' with command-line value: {cmd_value}")
                    setattr(opt, key, cmd_value)
                # else: # For cmd_opt specific args like 'defaults' itself
                      # setattr(opt, key, cmd_value) # This line can add 'defaults' to opt if needed, usually not.
    else:
        print("NOT using dataset default configurations primarily. Command-line arguments take precedence.")
        print("Any command-line argument left at its parser default will use the dataset's class default for that argument.")
        for key, default_class_value in vars(base_config).items():
            cmd_value = getattr(cmd_opt, key, None) # Get value from cmd_opt if it exists
            parser_default_value = parser.get_default(key)
            if cmd_value is not None and cmd_value != parser_default_value: # User provided a specific value for this key
                setattr(opt, key, cmd_value)
            # Else (user did not provide a specific value, or key not in cmd_opt), 'opt' already has 'default_class_value'
            
    # Final calculations for special-case parameters
    # Ensure ssl_projection_dim is correctly derived if set to indicator values
    if opt.ssl_projection_dim == -1: # -1 means use class default, which might be a value or calculated
        opt.ssl_projection_dim = base_config.ssl_projection_dim # Re-fetch from pristine base_config if needed
        if opt.ssl_projection_dim == 0 or opt.ssl_projection_dim == base_config.hiddenSize // 2 : # If class default was 0 or correctly calculated
             opt.ssl_projection_dim = opt.hiddenSize // 2
    elif opt.ssl_projection_dim == 0 : # If explicitly set to 0 via CMD (and not handled above)
        opt.ssl_projection_dim = opt.hiddenSize // 2

    # position_emb_dim: 0 means use hiddenSize
    if opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
        
    # max_len: 0 means it will be auto-detected in main() after data loading.
    # No change needed here, main() handles it.

    main(opt)
