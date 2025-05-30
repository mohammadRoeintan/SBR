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
    max_len = 50 # Max session length for position embedding, adjust based on dataset
    position_emb_dim = 100 # Should ideally be hiddenSize

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
    max_len = 50 # Max session length, adjust based on dataset
    position_emb_dim = 120 # Should ideally be hiddenSize


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
        print(f'Using validation set ({len(valid_data_raw[0])} sessions) for testing.')
    else:
        print(f'Using full test set ({len(test_data_raw[0])} sessions).')
    print(f'Training set size: {len(train_data_raw[0])} sessions.')


    # Initialize Dataset with opt to allow it to access opt.max_len for its internal self.len_max
    train_data_loader = Dataset(train_data_raw, shuffle=True, opt=opt)
    test_data_loader = Dataset(test_data_raw, shuffle=False, opt=opt)
    
    # Ensure opt.max_len is set to the actual max length found in the training data if not provided or too small
    # This is crucial for the Position Embedding layer size.
    actual_dataset_max_len = train_data_loader.len_max
    if opt.max_len < actual_dataset_max_len:
        print(f"Updating opt.max_len from {opt.max_len} to actual dataset max session length: {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    # Ensure position_emb_dim matches hiddenSize if addition is the combination strategy
    if opt.position_emb_dim != opt.hiddenSize:
        print(f"Adjusting opt.position_emb_dim from {opt.position_emb_dim} to opt.hiddenSize ({opt.hiddenSize}) for additive combination.")
        opt.position_emb_dim = opt.hiddenSize


    if opt.dataset == 'diginetica': n_node = 43098
    elif opt.dataset == 'yoochoose1_64': n_node = 37484 # Or yoochoose1_4
    else: n_node = 310; print(f"Warning: n_node using fallback {n_node}")

    model_instance = Attention_SessionGraph(opt, n_node)

    if opt.n_gpu > 1 and torch.cuda.is_available():
        print(f"Wrapping model with torch.nn.DataParallel for {opt.n_gpu} GPUs.")
        model = torch.nn.DataParallel(model_instance, device_ids=list(range(opt.n_gpu)))
        model.to(device) # Main model parts to cuda:0, DataParallel handles distribution
    else: model = model_instance.to(device) # Single GPU or CPU

    print(f"Model initialized. n_node={n_node}, max_session_len_for_pos_emb={opt.max_len}, pos_emb_dim={opt.position_emb_dim}")
    print(f"Hyperparameters: {vars(opt)}")

    start_time = time.time()
    best_result = [0.0, 0.0]; best_epoch = [0, 0]; bad_counter = 0

    for epoch_num in range(opt.epoch):
        print('-' * 50 + f'\nEpoch: {epoch_num}')
        # Pass the potentially wrapped model to train_test
        current_hit, current_mrr = train_test(model, train_data_loader, test_data_loader, opt, device)
        print(f'Epoch {epoch_num} Eval - Recall@20: {current_hit:.4f}, MRR@20: {current_mrr:.4f}')
        writer.add_scalar('epoch/recall_at_20', current_hit, epoch_num)
        writer.add_scalar('epoch/mrr_at_20', current_mrr, epoch_num)
        
        flag = 0
        saved_this_epoch_path = ""

        if current_hit > best_result[0] - 1e-5 : # Use a small epsilon for float comparison
            if abs(current_hit - best_result[0]) > 1e-5 or current_mrr > best_result[1] : # If recall is better, or recall same and mrr better
                best_result[0] = current_hit; best_epoch[0] = epoch_num; flag = 1
                saved_this_epoch_path = model_save_dir + f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt'
                state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_to_save, saved_this_epoch_path)
                print(f"Saved best model (by Recall) to {saved_this_epoch_path}")

        if current_mrr > best_result[1] - 1e-5 :
             if abs(current_mrr - best_result[1]) > 1e-5 or current_hit > best_result[0] : # If mrr is better, or mrr same and recall better
                best_result[1] = current_mrr; best_epoch[1] = epoch_num; flag = 1
                # Avoid saving twice if already saved for recall and path is identical
                mrr_save_path = model_save_dir + f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt'
                if mrr_save_path != saved_this_epoch_path:
                    state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                    torch.save(state_to_save, mrr_save_path)
                    print(f"Saved best model (by MRR) to {mrr_save_path}")
                elif not saved_this_epoch_path: # If not saved for recall but is best for MRR
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
    
    ds = parser.add_argument_group('Dataset Configuration')
    ds.add_argument('--dataset', default='diginetica', choices=['diginetica', 'yoochoose1_64'], help='Dataset name')
    ds.add_argument('--validation', type=str2bool, default=None, help='Use validation split (dataset default: True)')
    ds.add_argument('--valid_portion', type=float, default=-1.0, help='Validation split portion (dataset default: 0.1)')

    md = parser.add_argument_group('Model Hyperparameters')
    md.add_argument('--hiddenSize', type=int, default=0, help='Hidden state dimension (0 uses dataset default)')
    md.add_argument('--step', type=int, default=0, help='GNN propagation steps (0 uses dataset default)')
    md.add_argument('--nonhybrid', type=str2bool, default=None, help='Use non-hybrid scoring (dataset default: True)')
    md.add_argument('--max_len', type=int, default=0, help='Max session length for pos_emb (0 uses dataset default or auto-detect)')
    md.add_argument('--position_emb_dim', type=int, default=0, help='Dimension for position embeddings (0 uses hiddenSize)')


    tr = parser.add_argument_group('Training Hyperparameters')
    tr.add_argument('--defaults', type=str2bool, default=True, help='Use all default configurations for the chosen dataset')
    tr.add_argument('--n_gpu', type=int, default=0, help='Num GPUs (0: auto/CPU, >0: specific count)')
    tr.add_argument('--batchSize', type=int, default=0, help='Batch size (0 uses dataset default)')
    tr.add_argument('--epoch', type=int, default=0, help='Number of epochs (0 uses dataset default)')
    tr.add_argument('--lr', type=float, default=0.0, help='Learning rate (0.0 uses dataset default)')
    tr.add_argument('--lr_dc', type=float, default=0.0, help='LR decay rate (0.0 uses dataset default)')
    tr.add_argument('--lr_dc_step', type=int, default=0, help='Steps for LR decay (0 uses dataset default)')
    tr.add_argument('--l2', type=float, default=-1.0, help='L2 penalty (-1.0 uses dataset default)')
    tr.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 uses dataset default)')

    sl = parser.add_argument_group('Self-Supervised Learning (SSL) Hyperparameters')
    sl.add_argument('--ssl_weight', type=float, default=-1.0, help='SSL loss weight (-1.0 uses dataset default)')
    sl.add_argument('--ssl_temperature', type=float, default=-1.0, help='SSL InfoNCE temperature (-1.0 uses dataset default)')
    sl.add_argument('--ssl_item_drop_prob', type=float, default=-1.0, help='SSL item drop probability (-1.0 uses dataset default)')
    sl.add_argument('--ssl_projection_dim', type=int, default=-1, help='SSL projection head dim (-1 uses dataset default, 0 for hiddenSize/2)')

    cmd_opt = parser.parse_args()
    base_config = None

    if cmd_opt.dataset == 'diginetica': base_config = Diginetica_arg()
    elif cmd_opt.dataset == 'yoochoose1_64': base_config = Yoochoose_arg()
    else: print(f"FATAL: Unknown dataset '{cmd_opt.dataset}'"); sys.exit(1)

    # Initialize opt with base_config attributes
    opt = argparse.Namespace(**vars(base_config))

    # If --defaults is False, command-line args take precedence over class defaults for those args.
    # If --defaults is True, class defaults are used, but can be individually overridden by command-line args
    # if the command-line arg is *different* from its own parser default.
    
    # Create a merged opt: start with base_config, then override with CMD if CMD is not its own parser default
    merged_opt_vars = vars(base_config).copy()

    if cmd_opt.defaults:
        print("Using dataset default configurations. Explicit command-line arguments will override specific defaults.")
        for key, cmd_value in vars(cmd_opt).items():
            if cmd_value != parser.get_default(key): # If user provided a non-default CMD arg for this key
                if key in merged_opt_vars:
                     # print(f"Overriding default '{key}' from '{merged_opt_vars[key]}' with command-line value: {cmd_value}")
                     merged_opt_vars[key] = cmd_value
                # else:
                     # This could be for args like 'defaults' itself or if a new arg is added only to parser
                     # print(f"Note: CMD arg '{key}' (value: {cmd_value}) not in base config, using CMD value.")
                     # merged_opt_vars[key] = cmd_value # Add it if it's a valid setting we want to pass
    else: # Not using dataset defaults primarily. CMD args are king.
        print("NOT using dataset default configurations. Command-line arguments take precedence.")
        print("Any command-line argument left at its parser default will use the dataset's class default for that argument.")
        cmd_vars = vars(cmd_opt)
        for key_in_base in vars(base_config).keys():
            if key_in_base in cmd_vars : # If this base config key is also a cmd arg
                cmd_value_for_key = cmd_vars[key_in_base]
                if cmd_value_for_key != parser.get_default(key_in_base): # If user set it specifically
                    merged_opt_vars[key_in_base] = cmd_value_for_key
                # Else (user did not set it, it's at parser default), merged_opt_vars keeps the base_config value
            # Else (key_in_base not a cmd_arg), merged_opt_vars keeps the base_config value
        # Add any cmd_opt keys that were not in base_config (e.g. 'defaults' itself)
        for key_cmd, val_cmd in cmd_vars.items():
            if key_cmd not in merged_opt_vars:
                merged_opt_vars[key_cmd] = val_cmd


    opt = argparse.Namespace(**merged_opt_vars)

    # Final calculated/defaulted values
    if opt.ssl_projection_dim == -1 or (opt.ssl_projection_dim == 0 and base_config.ssl_projection_dim != 0): # -1 from parser means use class default, or if class default was not 0
        opt.ssl_projection_dim = opt.hiddenSize // 2
    elif opt.ssl_projection_dim == 0 and base_config.ssl_projection_dim == 0: # If class default was 0, means calculate
        opt.ssl_projection_dim = opt.hiddenSize // 2


    if opt.position_emb_dim == 0 : # 0 means use hiddenSize
        opt.position_emb_dim = opt.hiddenSize
    # opt.max_len will be refined in main() after loading data if it's still 0 or too small.

    main(opt)
