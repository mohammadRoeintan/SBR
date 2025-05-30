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
    n_gpu = 0 # Default: 0, meaning auto-detect or CPU
    max_len = 50 # Max session length for position embedding, adjust based on dataset
    position_emb_dim = 100 # Default: hiddenSize

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
    n_gpu = 0 # Default: 0, meaning auto-detect or CPU
    max_len = 50 # Max session length, adjust based on dataset
    position_emb_dim = 120 # Default: hiddenSize


def main(opt):
    model_save_dir = 'saved_ssl_time/'
    log_dir = 'logs_ssl_time/' # Ensure this is different or specific

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
        if not valid_data_raw[0]: print("Warning: Validation set is empty after split.")
        print(f'Using validation set ({len(valid_data_raw[0]) if valid_data_raw[0] else 0} sessions) for testing.')
    else:
        if not test_data_raw[0]: print("Warning: Test set is empty.")
        print(f'Using full test set ({len(test_data_raw[0]) if test_data_raw[0] else 0} sessions).')
    if not train_data_raw[0]: print("FATAL: Training set is empty."); sys.exit(1)
    print(f'Training set size: {len(train_data_raw[0])} sessions.')

    train_data_loader = Dataset(train_data_raw, shuffle=True, opt=opt)
    test_data_loader = Dataset(test_data_raw, shuffle=False, opt=opt)
    
    actual_dataset_max_len = train_data_loader.len_max
    if opt.max_len == 0 or opt.max_len < actual_dataset_max_len:
        print(f"Updating opt.max_len from {opt.max_len} to actual dataset max session length: {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    if opt.position_emb_dim == 0 : # If parser default 0 means use hiddenSize
        print(f"Adjusting opt.position_emb_dim from {opt.position_emb_dim} to opt.hiddenSize ({opt.hiddenSize}).")
        opt.position_emb_dim = opt.hiddenSize
        
    if opt.ssl_projection_dim == 0: # If parser default 0 means use hiddenSize // 2
        opt.ssl_projection_dim = opt.hiddenSize // 2
        print(f"Setting opt.ssl_projection_dim to hiddenSize // 2 = {opt.ssl_projection_dim}")


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
    
    # --- Grouped Arguments with specific defaults ---
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset', default='diginetica', choices=['diginetica', 'yoochoose1_64'], help='Dataset name')
    dataset_group.add_argument('--validation', type=str2bool, default=True, help='Use validation split')
    dataset_group.add_argument('--valid_portion', type=float, default=0.1, help='Validation split portion')

    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--hiddenSize', type=int, default=100, help='Hidden state dimension')
    model_group.add_argument('--step', type=int, default=1, help='GNN propagation steps')
    model_group.add_argument('--nonhybrid', type=str2bool, default=True, help='Use non-hybrid scoring')
    model_group.add_argument('--max_len', type=int, default=50, help='Max session length for pos_emb (can be auto-adjusted)')
    model_group.add_argument('--position_emb_dim', type=int, default=0, help='Dim for position embeddings (0 uses hiddenSize)') # Default 0

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--defaults', type=str2bool, default=True, help='Use all default configurations for the chosen dataset')
    train_group.add_argument('--n_gpu', type=int, default=0, help='Num GPUs (0: auto/CPU)') # Default 0
    train_group.add_argument('--batchSize', type=int, default=50, help='Batch size')
    train_group.add_argument('--epoch', type=int, default=30, help='Number of epochs')
    train_group.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_group.add_argument('--lr_dc', type=float, default=0.1, help='LR decay rate')
    train_group.add_argument('--lr_dc_step', type=int, default=3, help='Steps for LR decay')
    train_group.add_argument('--l2', type=float, default=1e-5, help='L2 penalty')
    train_group.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    ssl_group = parser.add_argument_group('Self-Supervised Learning (SSL) Hyperparameters')
    ssl_group.add_argument('--ssl_weight', type=float, default=0.1, help='SSL loss weight')
    ssl_group.add_argument('--ssl_temperature', type=float, default=0.07, help='SSL InfoNCE temperature')
    ssl_group.add_argument('--ssl_item_drop_prob', type=float, default=0.2, help='SSL item drop probability')
    ssl_group.add_argument('--ssl_projection_dim', type=int, default=0, help='SSL projection head dim (0 for hiddenSize/2)') # Default 0

    cmd_args = parser.parse_args()
    
    # --- Configuration Loading and Merging ---
    # Start with an empty Namespace or directly from cmd_args if not using class defaults as base
    opt = argparse.Namespace()

    if cmd_args.defaults:
        print("INFO: Using dataset default configurations as a base.")
        if cmd_args.dataset == 'diginetica':
            base_config = Diginetica_arg()
        elif cmd_args.dataset == 'yoochoose1_64':
            base_config = Yoochoose_arg()
        else:
            print(f"FATAL: Unknown dataset '{cmd_args.dataset}' for default configurations."); sys.exit(1)
        
        # Populate opt with base_config
        for key, value in vars(base_config).items():
            setattr(opt, key, value)
        
        # Override with any explicitly passed command-line args
        # An argument is considered "explicitly passed" if its value is different from
        # the default value defined in `parser.add_argument()`.
        for key, cmd_value in vars(cmd_args).items():
            if cmd_value != parser.get_default(key):
                # print(f"Overriding '{key}': from base_config '{getattr(opt, key, 'N/A')}' to CMD '{cmd_value}'")
                setattr(opt, key, cmd_value)
            elif not hasattr(opt, key): # If arg is not in base_config (e.g. 'defaults' itself)
                 setattr(opt, key, cmd_value)


    else: # Not using dataset defaults primarily. CMD args are king.
        print("INFO: NOT using dataset default configurations. Using command-line arguments directly.")
        # All values will come from cmd_args. The defaults in add_argument are used if user doesn't specify.
        opt = cmd_args

    # --- Post-processing for calculated/dependent defaults ---
    # This section is now called *after* opt is fully populated either from base_config+cmd_overrides
    # or directly from cmd_args.

    # If ssl_projection_dim is 0 (its parser default, or from class default that means "calculate")
    if opt.ssl_projection_dim == 0:
        if not hasattr(opt, 'hiddenSize'): # Should always be there if base_config was used or parser default used
             print("FATAL: hiddenSize not found in opt for calculating ssl_projection_dim. Check argument parsing.")
             sys.exit(1)
        opt.ssl_projection_dim = opt.hiddenSize // 2
        # print(f"INFO: Calculated ssl_projection_dim: {opt.ssl_projection_dim} (based on hiddenSize: {opt.hiddenSize})")

    # If position_emb_dim is 0 (its parser default, or from class default that means "use hiddenSize")
    if opt.position_emb_dim == 0:
        if not hasattr(opt, 'hiddenSize'):
             print("FATAL: hiddenSize not found in opt for calculating position_emb_dim. Check argument parsing.")
             sys.exit(1)
        opt.position_emb_dim = opt.hiddenSize
        # print(f"INFO: Set position_emb_dim to hiddenSize: {opt.position_emb_dim}")
        
    # opt.max_len will be refined in main() after loading data if it's still its initial default
    # from parser or class, and actual data has longer sequences.

    main(opt)
