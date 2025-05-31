import argparse
import pickle
import time
import sys
import os
import numpy as np
import torch
from proc_utils import Dataset, split_validation
from model import Attention_SessionGraph, train_test
from torch.utils.tensorboard import SummaryWriter

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

class Diginetica_arg:
    def __init__(self):
        self.dataset = 'diginetica'
        self.batchSize = 256
        self.hiddenSize = 512
        self.epoch = 100
        self.lr = 0.001
        self.l2 = 1e-4
        self.step = 3
        self.patience = 20
        self.nonhybrid = False
        self.validation = True
        self.valid_portion = 0.1
        self.ssl_weight = 0.5
        self.ssl_temperature = 0.1
        self.ssl_item_drop_prob = 0.4
        self.ssl_projection_dim = 256
        self.n_gpu = 1
        self.max_len = 0
        self.position_emb_dim = 0

class Yoochoose_arg:
    def __init__(self):
        self.dataset = 'yoochoose1_64'
        self.batchSize = 512
        self.hiddenSize = 512
        self.epoch = 100
        self.lr = 0.001
        self.l2 = 1e-4
        self.step = 4
        self.patience = 20
        self.nonhybrid = False
        self.validation = True
        self.valid_portion = 0.1
        self.ssl_weight = 0.5
        self.ssl_temperature = 0.1
        self.ssl_item_drop_prob = 0.4
        self.ssl_projection_dim = 256
        self.n_gpu = 1
        self.max_len = 0
        self.position_emb_dim = 0

def main(opt):
    data_dir = f'datasets/{opt.dataset}/'
    model_save_dir = 'saved_star_models/'
    log_dir = 'logs_star/'

    for directory in [model_save_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} ready.")

    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.n_gpu > 0 else "cpu")
    if torch.cuda.is_available() and opt.n_gpu > 0:
        print(f"Using {torch.cuda.device_count()} GPU(s)")
    else:
        print("Using CPU")

    try:
        with open(os.path.join(data_dir, 'train.txt'), 'rb') as f:
            train_data_raw = pickle.load(f)
        with open(os.path.join(data_dir, 'test.txt'), 'rb') as f:
            test_data_raw = pickle.load(f)
        with open(os.path.join(data_dir, 'time_data.pkl'), 'rb') as f:
            time_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if opt.validation:
        train_data_raw, valid_data_raw = split_validation(train_data_raw, opt.valid_portion)
        test_data_raw = valid_data_raw
        train_time_data, valid_time_data = split_validation(time_data, opt.valid_portion)
        time_data = train_time_data
        print(f'Using validation set ({len(valid_data_raw[0])} sessions) for testing.')
    else:
        print(f'Using full test set ({len(test_data_raw[0])} sessions).')
        
    print(f'Training set size: {len(train_data_raw[0])} sessions.')

    train_data_loader = Dataset(train_data_raw, time_data, shuffle=True, opt=opt)
    test_data_loader = Dataset(test_data_raw, time_data, shuffle=False, opt=opt)
    
    actual_dataset_max_len = train_data_loader.len_max
    if opt.max_len == 0 or opt.max_len < actual_dataset_max_len:
        print(f"Updating opt.max_len from {opt.max_len} to {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    if opt.position_emb_dim == 0: 
        opt.position_emb_dim = opt.hiddenSize
        
    if opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2

    if opt.dataset == 'diginetica': 
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64': 
        n_node = 37484
    else: 
        all_items = set()
        for session in train_data_raw[0] + test_data_raw[0]:
            all_items.update(session)
        n_node = len(all_items) + 1
        print(f"Calculated n_node: {n_node}")

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
            saved_this_epoch_path = os.path.join(model_save_dir, f'best_recall_epoch_{epoch_num}.pt')
            state_to_save = {
                'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'opt': opt,
                'epoch': epoch_num,
                'recall': current_hit,
                'mrr': current_mrr
            }
            torch.save(state_to_save, saved_this_epoch_path)
            print(f"Saved best model (by Recall) to {saved_this_epoch_path}")

        if current_mrr >= best_result[1]:
            best_result[1] = current_mrr
            best_epoch[1] = epoch_num
            flag = 1
            mrr_save_path = os.path.join(model_save_dir, f'best_mrr_epoch_{epoch_num}.pt')
            if mrr_save_path != saved_this_epoch_path:
                state_to_save = {
                    'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'opt': opt,
                    'epoch': epoch_num,
                    'recall': current_hit,
                    'mrr': current_mrr
                }
                torch.save(state_to_save, mrr_save_path)
                print(f"Saved best model (by MRR) to {mrr_save_path}")

        print(f'Current Best: Recall@20: {best_result[0]:.4f} (Epoch {best_epoch[0]}), MRR@20: {best_result[1]:.4f} (Epoch {best_epoch[1]})')
        
        bad_counter = 0 if flag else bad_counter + 1
        if bad_counter >= opt.patience:
            print(f"Early stopping after {opt.patience} epochs without improvement.")
            break
            
    writer.close()
    print('-' * 50 + f"\nTotal Running time: {(time.time() - start_time)/3600:.2f} hours")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', default='yoochoose1_64', choices=['diginetica', 'yoochoose1_64'], help='Dataset name')
    parser.add_argument('--validation', type=str2bool, default=True, help='Use validation split')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='Validation split portion')

    parser.add_argument('--hiddenSize', type=int, default=512, help='Hidden state dimension')
    parser.add_argument('--step', type=int, default=4, help='GNN propagation steps')
    parser.add_argument('--nonhybrid', type=str2bool, default=False, help='Use non-hybrid scoring')
    parser.add_argument('--max_len', type=int, default=0, help='Max session length (0=auto)')
    parser.add_argument('--position_emb_dim', type=int, default=0, help='Position embedding dim (0=hiddenSize)')

    parser.add_argument('--n_gpu', type=int, default=1, help='Num GPUs (0=CPU)')
    parser.add_argument('--batchSize', type=int, default=512, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2 penalty')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    parser.add_argument('--ssl_weight', type=float, default=0.5, help='SSL loss weight')
    parser.add_argument('--ssl_temperature', type=float, default=0.1, help='SSL temperature')
    parser.add_argument('--ssl_item_drop_prob', type=float, default=0.4, help='Item dropout prob')
    parser.add_argument('--ssl_projection_dim', type=int, default=0, help='Projection dim (0=hiddenSize/2)')

    cmd_args = parser.parse_args()
    
    opt = argparse.Namespace()
    
    if cmd_args.dataset == 'diginetica':
        base_config = Diginetica_arg()
    elif cmd_args.dataset == 'yoochoose1_64':
        base_config = Yoochoose_arg()
    else:
        print(f"Error: Unknown dataset '{cmd_args.dataset}'")
        sys.exit(1)
    
    for key, value in vars(base_config).items():
        setattr(opt, key, value)
    
    for key, value in vars(cmd_args).items():
        if hasattr(opt, key):
            setattr(opt, key, value)
    
    if opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2

    if opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
        
    print("Final configuration:")
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    
    main(opt)
