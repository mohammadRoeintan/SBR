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
        print(f'Using validation set ({len(valid_data_raw[0])} sessions) for testing.')
    else:
        print(f'Using full test set ({len(test_data_raw[0])} sessions).')
    print(f'Training set size: {len(train_data_raw[0])} sessions.')

    train_data_loader = Dataset(train_data_raw, shuffle=True, opt=opt)
    test_data_loader = Dataset(test_data_raw, shuffle=False, opt=opt)
    
    actual_dataset_max_len = train_data_loader.len_max
    if opt.max_len < actual_dataset_max_len:
        print(f"Updating opt.max_len from {opt.max_len} to actual dataset max session length: {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    if opt.position_emb_dim != opt.hiddenSize and opt.position_emb_dim == base_config.position_emb_dim : # if it's still the class default and not equal
        print(f"Adjusting opt.position_emb_dim from {opt.position_emb_dim} to opt.hiddenSize ({opt.hiddenSize}) for additive combination.")
        opt.position_emb_dim = opt.hiddenSize

    if opt.ssl_projection_dim == -1 or (opt.ssl_projection_dim == 0 and base_config.ssl_projection_dim != 0) :
         opt.ssl_projection_dim = opt.hiddenSize // 2
    elif opt.ssl_projection_dim == 0 and base_config.ssl_projection_dim == 0:
         opt.ssl_projection_dim = opt.hiddenSize // 2


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
    print(f"Final Hyperparameters for model: {vars(opt)}")

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
            if abs(current_hit - best_result[0]) > 1e-5 or current_mrr > best_result[1] :
                best_result[0] = current_hit; best_epoch[0] = epoch_num; flag = 1
                saved_this_epoch_path = model_save_dir + f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt'
                state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_to_save, saved_this_epoch_path)
                print(f"Saved best model (by Recall) to {saved_this_epoch_path}")

        if current_mrr > best_result[1] - 1e-5 :
             if abs(current_mrr - best_result[1]) > 1e-5 or current_hit > best_result[0] :
                best_result[1] = current_mrr; best_epoch[1] = epoch_num; flag = 1
                mrr_save_path = model_save_dir + f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt'
                if mrr_save_path != saved_this_epoch_path:
                    state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                    torch.save(state_to_save, mrr_save_path)
                    print(f"Saved best model (by MRR) to {mrr_save_path}")
                elif not saved_this_epoch_path:
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
    ds_group.add_argument('--validation', type=str2bool, default=None, metavar='BOOL', help='Use validation split (dataset default: True)')
    ds_group.add_argument('--valid_portion', type=float, default=None, metavar='FLOAT', help='Validation split portion (dataset default: 0.1)')

    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--hiddenSize', type=int, default=None, metavar='INT', help='Hidden state dimension (dataset default)')
    model_group.add_argument('--step', type=int, default=None, metavar='INT', help='GNN propagation steps (dataset default)')
    model_group.add_argument('--nonhybrid', type=str2bool, default=None, metavar='BOOL', help='Use non-hybrid scoring (dataset default: True)')
    model_group.add_argument('--max_len', type=int, default=None, metavar='INT', help='Max session length for pos_emb (dataset default or auto-detect)')
    model_group.add_argument('--position_emb_dim', type=int, default=None, metavar='INT', help='Dimension for position embeddings (dataset default: hiddenSize)')

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--defaults', type=str2bool, default=True, metavar='BOOL', help='Use all default configurations for the chosen dataset')
    train_group.add_argument('--n_gpu', type=int, default=None, metavar='INT', help='Num GPUs (dataset default: 0, meaning auto/CPU)')
    train_group.add_argument('--batchSize', type=int, default=None, metavar='INT', help='Batch size (dataset default)')
    train_group.add_argument('--epoch', type=int, default=None, metavar='INT', help='Number of epochs (dataset default)')
    train_group.add_argument('--lr', type=float, default=None, metavar='FLOAT', help='Learning rate (dataset default)')
    train_group.add_argument('--lr_dc', type=float, default=None, metavar='FLOAT', help='LR decay rate (dataset default)')
    train_group.add_argument('--lr_dc_step', type=int, default=None, metavar='INT', help='Steps for LR decay (dataset default)')
    train_group.add_argument('--l2', type=float, default=None, metavar='FLOAT', help='L2 penalty (dataset default)')
    train_group.add_argument('--patience', type=int, default=None, metavar='INT', help='Early stopping patience (dataset default)')

    ssl_group = parser.add_argument_group('Self-Supervised Learning (SSL) Hyperparameters')
    ssl_group.add_argument('--ssl_weight', type=float, default=None, metavar='FLOAT', help='SSL loss weight (dataset default)')
    ssl_group.add_argument('--ssl_temperature', type=float, default=None, metavar='FLOAT', help='SSL InfoNCE temperature (dataset default)')
    ssl_group.add_argument('--ssl_item_drop_prob', type=float, default=None, metavar='FLOAT', help='SSL item drop probability (dataset default)')
    ssl_group.add_argument('--ssl_projection_dim', type=int, default=None, metavar='INT', help='SSL projection head dim (dataset default: hiddenSize/2)')

    cmd_args = parser.parse_args()
    base_config = None

    if cmd_args.dataset == 'diginetica': base_config = Diginetica_arg()
    elif cmd_args.dataset == 'yoochoose1_64': base_config = Yoochoose_arg()
    else: print(f"FATAL: Unknown dataset '{cmd_args.dataset}'"); sys.exit(1)

    # Initialize final_opt with a copy of the base configuration for the selected dataset
    final_opt_vars = vars(base_config).copy()

    if cmd_args.defaults:
        print("INFO: Using dataset default configurations. Explicit command-line arguments will override specific defaults.")
        # Override specific defaults if a command-line argument was explicitly provided (i.e., not None)
        for key, cmd_value in vars(cmd_args).items():
            if cmd_value is not None and key != 'defaults': # 'defaults' arg itself is not a config
                if key in final_opt_vars:
                    # print(f"Overriding '{key}': from '{final_opt_vars[key]}' to CMD '{cmd_value}'")
                    final_opt_vars[key] = cmd_value
                else:
                    # This case should ideally not happen if all parser args are in base_config classes
                    # print(f"Warning: CMD arg '{key}' not in base_config class, adding it.")
                    final_opt_vars[key] = cmd_value

    else: # Not using dataset defaults primarily. CMD args are king.
        print("INFO: NOT using all dataset defaults. Command-line arguments take precedence.")
        print("      Any command-line argument NOT provided will use the dataset's class default for that argument.")
        cmd_vars = vars(cmd_args)
        for key_in_base, base_value in vars(base_config).items():
            if key_in_base in cmd_vars: # If this base config key is also a cmd arg
                cmd_value_for_key = cmd_vars[key_in_base]
                if cmd_value_for_key is not None: # If user set it specifically via CMD
                    final_opt_vars[key_in_base] = cmd_value_for_key
                # Else (user did not set it, it's None), final_opt_vars keeps the base_config value (already there)
            # Else (key_in_base not a cmd_arg), final_opt_vars keeps the base_config value

        # Add any cmd_args that were not in base_config (e.g., 'defaults' itself)
        for key_cmd, val_cmd in cmd_vars.items():
            if key_cmd not in final_opt_vars and val_cmd is not None : # Also check val_cmd is not None
                final_opt_vars[key_cmd] = val_cmd


    opt = argparse.Namespace(**final_opt_vars)

    # --- Post-processing and defaulting calculated values ---
    # Ensure these critical calculated fields are set if they relied on a 'None' or sentinel from parser
    # or if their base_config value indicates they should be calculated.
    
    # SSL Projection Dimension
    if opt.ssl_projection_dim is None or opt.ssl_projection_dim == 0 : # 0 was a special value in prev logic, None is from parser default now
        opt.ssl_projection_dim = opt.hiddenSize // 2
        print(f"INFO: Setting ssl_projection_dim to hiddenSize//2 = {opt.ssl_projection_dim}")

    # Position Embedding Dimension
    if opt.position_emb_dim is None or opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
        print(f"INFO: Setting position_emb_dim to hiddenSize = {opt.position_emb_dim}")
        
    # max_len will be refined in main() after loading data if it's still the class default (e.g. 50)
    # and actual data has longer sequences.

    main(opt)
