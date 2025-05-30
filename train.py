import argparse
import pickle
import time
import sys
import os

import torch
from proc_utils import Dataset, split_validation
from model import Attention_SessionGraph, train_test
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    return v.lower() in ('true')

class Diginetica_arg():
    dataset = 'diginetica'
    batchSize = 100
    hiddenSize = 150
    epoch = 50
    lr = 0.002
    lr_dc = 0.2
    lr_dc_step = 5
    l2 = 2e-5
    step = 2
    patience = 15
    nonhybrid = True
    validation = True
    valid_portion = 0.1
    ssl_weight = 0.25
    ssl_temperature = 0.12
    ssl_item_drop_prob = 0.35
    ssl_projection_dim = 75
    n_gpu = 0
    max_len = 50
    position_emb_dim = 150

class Yoochoose_arg():
    dataset = 'yoochoose1_64'
    batchSize = 100
    hiddenSize = 150
    epoch = 50
    lr = 0.002
    lr_dc = 0.2
    lr_dc_step = 5
    l2 = 2e-5
    step = 2
    patience = 15
    nonhybrid = True
    validation = True
    valid_portion = 0.1
    ssl_weight = 0.25
    ssl_temperature = 0.12
    ssl_item_drop_prob = 0.35
    ssl_projection_dim = 75
    n_gpu = 0
    max_len = 50
    position_emb_dim = 150


def main(opt):
    model_save_dir = 'saved_ssl_time/'
    log_dir = 'logs_ssl_time/'

    for directory in [model_save_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")

    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.n_gpu > 0 else "cpu")
    if torch.cuda.is_available() and opt.n_gpu > 0:
        print(f"Using {opt.n_gpu} GPU(s)")
    else:
        print("Using CPU")

    if opt.dataset == 'diginetica':
        train_data_path = 'datasets/cikm16/raw/train.txt'
        test_data_path = 'datasets/cikm16/raw/test.txt'
    elif opt.dataset == 'yoochoose1_64':
        train_data_path = 'datasets/yoochoose1_64/raw/train.txt'
        test_data_path = 'datasets/yoochoose1_64/raw/test.txt'
    else: 
        print(f"Error: Unknown dataset {opt.dataset}")
        sys.exit(1)
        
    try:
        with open(train_data_path, 'rb') as f:
            train_data_raw = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            test_data_raw = pickle.load(f)
    except FileNotFoundError: 
        print(f"Error: Dataset file not found at {train_data_path} or {test_data_path}")
        sys.exit(1)

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
    if opt.max_len == 0 or opt.max_len < actual_dataset_max_len:
        print(f"Updating opt.max_len from {opt.max_len} to actual dataset max session length: {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    if opt.position_emb_dim == 0 : 
        opt.position_emb_dim = opt.hiddenSize
        
    if opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2

    if opt.dataset == 'diginetica': 
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64': 
        n_node = 37484
    else: 
        n_node = 310

    model_instance = Attention_SessionGraph(opt, n_node)

    if opt.n_gpu > 1 and torch.cuda.is_available():
        print(f"Using {opt.n_gpu} GPUs with DataParallel")
        model = torch.nn.DataParallel(model_instance)
    else:
        model = model_instance
        
    model = model.to(device)

    print(f"Model initialized. n_node={n_node}, max_session_len={opt.max_len}, pos_emb_dim={opt.position_emb_dim}")
    print(f"Hyperparameters: {vars(opt)}")

    start_time = time.time()
    best_result = [0.0, 0.0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch_num in range(opt.epoch):
        print('-' * 50 + f'\nEpoch: {epoch_num}')
        opt.current_epoch_num = epoch_num
        current_hit, current_mrr = train_test(model, train_data_loader, test_data_loader, opt, device)
        print(f'Epoch {epoch_num} Eval - Recall@20: {current_hit:.4f}, MRR@20: {current_mrr:.4f}')
        writer.add_scalar('epoch/recall_at_20', current_hit, epoch_num)
        writer.add_scalar('epoch/mrr_at_20', current_mrr, epoch_num)
        
        flag = 0
        saved_this_epoch_path = ""

        if current_hit >= best_result[0]:
            best_result[0] = current_hit
            best_epoch[0] = epoch_num
            flag = 1
            saved_this_epoch_path = os.path.join(model_save_dir, f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt')
            state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, saved_this_epoch_path)
            print(f"Saved best model (by Recall) to {saved_this_epoch_path}")

        if current_mrr >= best_result[1]:
            best_result[1] = current_mrr
            best_epoch[1] = epoch_num
            flag = 1
            mrr_save_path = os.path.join(model_save_dir, f'epoch_{epoch_num}_recall_{current_hit:.4f}_mrr_{current_mrr:.4f}.pt')
            if mrr_save_path != saved_this_epoch_path:
                state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_to_save, mrr_save_path)
                print(f"Saved best model (by MRR) to {mrr_save_path}")

        print(f'Current Best: Recall@20: {best_result[0]:.4f} (Epoch {best_epoch[0]}), MRR@20: {best_result[1]:.4f} (Epoch {best_epoch[1]})')
        bad_counter = 0 if flag else bad_counter + 1
        if bad_counter >= opt.patience:
            print(f"Early stopping after {opt.patience} epochs without improvement.")
            break
            
    writer.close()
    print('-' * 50 + f"\nTotal Running time: {(time.time() - start_time):.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument('--dataset', default='diginetica', choices=['diginetica', 'yoochoose1_64'], help='Dataset name')
    dataset_group.add_argument('--validation', type=str2bool, default=True, help='Use validation split')
    dataset_group.add_argument('--valid_portion', type=float, default=0.1, help='Validation split portion')

    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--hiddenSize', type=int, default=150, help='Hidden state dimension')
    model_group.add_argument('--step', type=int, default=2, help='GNN propagation steps')
    model_group.add_argument('--nonhybrid', type=str2bool, default=True, help='Use non-hybrid scoring')
    model_group.add_argument('--max_len', type=int, default=50, help='Max session length for pos_emb (can be auto-adjusted)')
    model_group.add_argument('--position_emb_dim', type=int, default=0, help='Dim for position embeddings (0 uses hiddenSize)')

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--defaults', type=str2bool, default=True, help='Use all default configurations for the chosen dataset')
    train_group.add_argument('--n_gpu', type=int, default=0, help='Num GPUs (0: auto/CPU)')
    train_group.add_argument('--batchSize', type=int, default=100, help='Batch size')
    train_group.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    train_group.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    train_group.add_argument('--lr_dc', type=float, default=0.2, help='LR decay rate')
    train_group.add_argument('--lr_dc_step', type=int, default=5, help='Steps for LR decay')
    train_group.add_argument('--l2', type=float, default=2e-5, help='L2 penalty')
    train_group.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    ssl_group = parser.add_argument_group('Self-Supervised Learning (SSL) Hyperparameters')
    ssl_group.add_argument('--ssl_weight', type=float, default=0.25, help='SSL loss weight')
    ssl_group.add_argument('--ssl_temperature', type=float, default=0.12, help='SSL InfoNCE temperature')
    ssl_group.add_argument('--ssl_item_drop_prob', type=float, default=0.35, help='SSL item drop probability')
    ssl_group.add_argument('--ssl_projection_dim', type=int, default=0, help='SSL projection head dim (0 for hiddenSize/2)')

    cmd_args = parser.parse_args()
    
    opt = argparse.Namespace()

    if cmd_args.defaults:
        if cmd_args.dataset == 'diginetica':
            base_config = Diginetica_arg()
        elif cmd_args.dataset == 'yoochoose1_64':
            base_config = Yoochoose_arg()
        else:
            print(f"FATAL: Unknown dataset '{cmd_args.dataset}'")
            sys.exit(1)
        
        for key, value in vars(base_config).items():
            setattr(opt, key, value)
        
        for key, cmd_value in vars(cmd_args).items():
            if cmd_value != parser.get_default(key):
                setattr(opt, key, cmd_value)
            elif not hasattr(opt, key):
                 setattr(opt, key, cmd_value)
    else:
        opt = cmd_args

    if not hasattr(opt, 'ssl_projection_dim') or opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2

    if not hasattr(opt, 'position_emb_dim') or opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
        
    if not hasattr(opt, 'n_gpu'):
        opt.n_gpu = 0
        
    if not hasattr(opt, 'max_len'):
        opt.max_len = 50

    main(opt)
