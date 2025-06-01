import argparse
import pickle
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from proc_utils import Dataset, split_validation #
from model import Attention_SessionGraph, train_test #
from torch.utils.tensorboard import SummaryWriter #

def str2bool(v):
    return v.lower() in ('true', '1', 'yes') #

class Diginetica_arg:
    def __init__(self):
        self.dataset = 'diginetica' #
        self.batchSize = 256 #
        self.hiddenSize = 512 #
        self.epoch = 100 #
        self.lr = 0.001 #
        self.l2 = 1e-4 #
        self.step = 3 #
        self.patience = 20 #
        self.nonhybrid = False #
        self.validation = True #
        self.valid_portion = 0.1 #
        self.ssl_weight = 0.5 #
        self.ssl_temperature = 0.1 #
        self.ssl_item_drop_prob = 0.4 #
        self.ssl_projection_dim = 256 #
        self.n_gpu = 1 #
        self.max_len = 0 #
        self.position_emb_dim = 0 #

class Yoochoose_arg:
    def __init__(self):
        self.dataset = 'yoochoose1_64' #
        self.batchSize = 512 #
        self.hiddenSize = 512 #
        self.epoch = 100 #
        self.lr = 0.001 #
        self.l2 = 1e-4 #
        self.step = 4 #
        self.patience = 20 #
        self.nonhybrid = False #
        self.validation = True #
        self.valid_portion = 0.1 #
        self.ssl_weight = 0.5 #
        self.ssl_temperature = 0.1 #
        self.ssl_item_drop_prob = 0.4 #
        self.ssl_projection_dim = 256 #
        self.n_gpu = 1 #
        self.max_len = 0 #
        self.position_emb_dim = 0 #

def get_n_node_from_data(train_data_raw, test_data_raw_orig):
    max_item_id = 0
    for seq_list in [train_data_raw[0], test_data_raw_orig[0]]:
        for seq in seq_list:
            for item_id in seq:
                if item_id > max_item_id:
                    max_item_id = item_id
    return max_item_id + 1 # +1 for padding index 0

# تابع جدید برای ارزیابی نهایی روی مجموعه تست
def evaluate_final_test(model, test_data_loader, opt, device):
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    print(f'Start Final Test Set Prediction/Evaluation: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    model.eval()
    hit, mrr = [], []
    
    slices_test = test_data_loader.generate_batch(opt.batchSize)
    
    if not slices_test:
        print("Warning: No batches generated from final test data. Skipping final test evaluation.")
        return 0.0, 0.0

    with torch.no_grad():
        for i_test_slice_indices in slices_test: # tqdm can be added here if desired
            if len(i_test_slice_indices) == 0:
                continue
            
            data_v1_test_tuple, _, targets_test_np, mask_test_np, time_diffs_test_np, _ = test_data_loader.get_slice(
                i_test_slice_indices, ssl_item_drop_prob=0.0 # No dropout for eval
            )

            alias_inputs_test, A_test, items_test_unique, _, position_ids_test = data_v1_test_tuple

            if items_test_unique.size == 0:
                print("Skipping final test batch due to empty unique items array.")
                continue

            items_test_unique = torch.from_numpy(items_test_unique).long().to(device)
            A_test = torch.from_numpy(A_test).float().to(device)
            alias_inputs_test = torch.from_numpy(alias_inputs_test).long().to(device)
            position_ids_test = torch.from_numpy(position_ids_test).long().to(device)
            time_diffs_test = torch.from_numpy(time_diffs_test_np).float().to(device)

            mask_test_cuda = torch.from_numpy(mask_test_np).long().to(device)
            targets_test_cuda = torch.from_numpy(targets_test_np).long().to(device)
            
            seq_attention_mask_test = (mask_test_cuda == 0) 

            final_seq_hidden_test, star_context_test = model(items_test_unique, A_test, alias_inputs_test, position_ids_test, time_diffs_test, seq_attention_mask_test)

            if final_seq_hidden_test.size(0) == 0:
                print("Skipping final test batch due to empty final_seq_hidden_test.")
                continue

            scores_test = model_module.compute_scores(final_seq_hidden_test, star_context_test, mask_test_cuda)

            sub_scores_top20_indices = scores_test.topk(20)[1] 
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            
            targets_test_for_metric_np = targets_test_cuda.cpu().detach().numpy() - 1 

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_test_for_metric_np):
                hit.append(np.isin(target_item, score_row))
                if target_item in score_row:
                    rank = np.where(score_row == target_item)[0][0] + 1 
                    mrr.append(1.0 / rank)
                else:
                    mrr.append(0.0)

    test_hit_metric = np.mean(hit) * 100 if hit else 0.0
    test_mrr_metric = np.mean(mrr) * 100 if mrr else 0.0
    
    return test_hit_metric, test_mrr_metric

def main(opt):
    data_dir = f'datasets/{opt.dataset}/' #
    model_save_dir = 'saved_star_models/' #
    log_dir = 'logs_star/' #
    plot_dir = 'plots_star/' # New directory for plots

    for directory in [model_save_dir, log_dir, plot_dir]: #
        if not os.path.exists(directory):
            os.makedirs(directory) #
            print(f"Directory {directory} created.")

    writer = SummaryWriter(log_dir=log_dir) #

    device = torch.device("cuda" if torch.cuda.is_available() and opt.n_gpu > 0 else "cpu") #
    if torch.cuda.is_available() and opt.n_gpu > 0: #
        print(f"Using {torch.cuda.device_count()} GPU(s)") #
    else:
        print("Using CPU") #

    try:
        with open(os.path.join(data_dir, 'train.txt'), 'rb') as f: #
            train_data_raw = pickle.load(f) #
        with open(os.path.join(data_dir, 'test.txt'), 'rb') as f: #
            test_data_raw_orig = pickle.load(f) #
        with open(os.path.join(data_dir, 'time_data.pkl'), 'rb') as f: #
            time_data_all = pickle.load(f) #
    except Exception as e:
        print(f"Error loading data: {e}") #
        sys.exit(1) #

    time_data_train_orig = time_data_all['train_time_diffs'] #
    time_data_test_orig = time_data_all['test_time_diffs'] #

    n_node = get_n_node_from_data(train_data_raw, test_data_raw_orig)
    print(f"Dynamically calculated n_node: {n_node}")

    train_loss_history = []
    train_main_loss_history = []
    train_ssl_loss_history = []
    val_recall_history = []
    val_mrr_history = []

    # مسیر بهترین مدل ذخیره شده بر اساس Recall
    best_recall_model_path = ""

    if opt.validation:
        (train_seqs, train_targets), (valid_seqs, valid_targets) = split_validation(
            train_data_raw, opt.valid_portion #
        )
        train_data_raw_for_train = (train_seqs, train_targets) # تغییر نام برای وضوح
        eval_data_raw = (valid_seqs, valid_targets) # داده اعتبارسنجی
        
        n_samples = len(time_data_train_orig) #
        indices = list(range(n_samples)) #
        np.random.shuffle(indices) #
        n_train = int(np.round(n_samples * (1. - opt.valid_portion))) #
        
        time_data_train_for_train = [time_data_train_orig[i] for i in indices[:n_train]] # زمان برای داده آموزش
        time_data_for_eval = [time_data_train_orig[i] for i in indices[n_train:]] # زمان برای داده اعتبارسنجی
    else:
        train_data_raw_for_train = train_data_raw # اگر اعتبارسنجی نباشد، همه داده آموزش برای آموزش است
        eval_data_raw = test_data_raw_orig # ارزیابی روی داده تست اصلی انجام می‌شود
        time_data_train_for_train = time_data_train_orig
        time_data_for_eval = time_data_test_orig

    train_data_loader = Dataset(train_data_raw_for_train, time_data_train_for_train, shuffle=True, opt=opt) #
    eval_data_loader = Dataset(eval_data_raw, time_data_for_eval, shuffle=False, opt=opt) #
    
    # استفاده از نام متغیر صحیح که در کلاس Dataset تعریف شده
    actual_dataset_max_len = train_data_loader.len_max_for_padding #
    if opt.max_len == 0 or (actual_dataset_max_len > 0 and opt.max_len < actual_dataset_max_len) : #
        print(f"Updating opt.max_len from {opt.max_len} to {actual_dataset_max_len}") #
        opt.max_len = actual_dataset_max_len #
    
    if opt.position_emb_dim == 0:  #
        opt.position_emb_dim = opt.hiddenSize #
        
    if opt.ssl_projection_dim == 0: #
        opt.ssl_projection_dim = opt.hiddenSize // 2 #

    model_instance = Attention_SessionGraph(opt, n_node) #

    if opt.n_gpu > 1 and torch.cuda.is_available(): #
        print(f"Using {opt.n_gpu} GPUs with DataParallel") #
        model = torch.nn.DataParallel(model_instance) #
    else:
        model = model_instance #
        
    model = model.to(device) #

    print(f"Model initialized. n_node={n_node}, max_session_len={opt.max_len}, pos_emb_dim={opt.position_emb_dim}") #
    print(f"Hyperparameters: {vars(opt)}") #

    start_time = time.time() #
    best_result_val = [0.0, 0.0] # بهترین نتایج اعتبارسنجی
    best_epoch_val = [0, 0] # بهترین اپک‌های اعتبارسنجی
    bad_counter = 0 #

    for epoch_num in range(opt.epoch): #
        print('-' * 50 + f'\nEpoch: {epoch_num}') #
        opt.current_epoch_num = epoch_num #
        
        eval_hit, eval_mrr, avg_train_total_loss, avg_train_main_loss, avg_train_ssl_loss = train_test(
            model, train_data_loader, eval_data_loader, opt, device #
        )

        print(f'Epoch {epoch_num} Validation - Recall@20: {eval_hit:.4f}, MRR@20: {eval_mrr:.4f}') #
        print(f'Epoch {epoch_num} Train Avg Loss - Total: {avg_train_total_loss:.4f}, Main: {avg_train_main_loss:.4f}, SSL: {avg_train_ssl_loss:.4f}')

        writer.add_scalar('epoch/eval_recall_at_20', eval_hit, epoch_num) #
        writer.add_scalar('epoch/eval_mrr_at_20', eval_mrr, epoch_num) #
        writer.add_scalar('epoch/train_total_loss', avg_train_total_loss, epoch_num)
        writer.add_scalar('epoch/train_main_loss', avg_train_main_loss, epoch_num)
        writer.add_scalar('epoch/train_ssl_loss', avg_train_ssl_loss, epoch_num)
        
        train_loss_history.append(avg_train_total_loss)
        train_main_loss_history.append(avg_train_main_loss)
        train_ssl_loss_history.append(avg_train_ssl_loss)
        val_recall_history.append(eval_hit)
        val_mrr_history.append(eval_mrr)

        flag = 0 
        current_epoch_recall_model_path = "" 

        if eval_hit >= best_result_val[0]: 
            best_result_val[0] = eval_hit 
            best_epoch_val[0] = epoch_num 
            flag = 1 
            # ذخیره مسیر بهترین مدل بر اساس Recall
            best_recall_model_path = os.path.join(model_save_dir, f'best_RECALL_model_epoch_{epoch_num}_recall_{eval_hit:.4f}_mrr_{eval_mrr:.4f}.pt') 
            current_epoch_recall_model_path = best_recall_model_path
            state_to_save = { 
                'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), 
                'opt': opt, 
                'epoch': epoch_num, 
                'recall': eval_hit, 
                'mrr': eval_mrr 
            }
            torch.save(state_to_save, best_recall_model_path) 
            print(f"Saved best model (by Validation Recall) to {best_recall_model_path}") 

        if eval_mrr >= best_result_val[1]: 
            best_result_val[1] = eval_mrr 
            best_epoch_val[1] = epoch_num 
            flag = 1 
            mrr_save_path = os.path.join(model_save_dir, f'best_MRR_model_epoch_{epoch_num}_recall_{eval_hit:.4f}_mrr_{eval_mrr:.4f}.pt') 
            if mrr_save_path != current_epoch_recall_model_path: 
                state_to_save = { 
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), 
                    'opt': opt, 
                    'epoch': epoch_num, 
                    'recall': eval_hit, 
                    'mrr': eval_mrr 
                }
                torch.save(state_to_save, mrr_save_path) 
                print(f"Saved best model (by Validation MRR) to {mrr_save_path}") 

        print(f'Current Best Validation: Recall@20: {best_result_val[0]:.4f} (Epoch {best_epoch_val[0]}), MRR@20: {best_result_val[1]:.4f} (Epoch {best_epoch_val[1]})') #
        
        bad_counter = 0 if flag else bad_counter + 1 #
        if bad_counter >= opt.patience: #
            print(f"Early stopping after {opt.patience} epochs without improvement.") #
            break #
            
    writer.close() #
    print('-' * 50 + f"\nTotal Training and Validation Running time: {(time.time() - start_time)/3600:.2f} hours") #

    # --- ارزیابی نهایی روی مجموعه تست با بهترین مدل (بر اساس Recall) ---
    if best_recall_model_path and os.path.exists(best_recall_model_path):
        print("\n" + "="*50)
        print(f"Loading best model for final test evaluation from: {best_recall_model_path}")
        # اضافه کردن weights_only=False برای بارگذاری صحیح
        checkpoint = torch.load(best_recall_model_path, map_location=device, weights_only=False)
        
        opt_from_checkpoint = checkpoint['opt']
        # هنگام ایجاد مدل برای تست، باید از n_node که در ابتدای تابع main محاسبه شده استفاده کنیم.
        # opt_from_checkpoint ممکن است n_node نداشته باشد یا مقدار متفاوتی داشته باشد اگر از اجرای قبلی ذخیره شده باشد.
        model_for_test_instance = Attention_SessionGraph(opt_from_checkpoint, n_node) 

        if opt_from_checkpoint.n_gpu > 1 and torch.cuda.is_available(): 
             model_for_test = torch.nn.DataParallel(model_for_test_instance)
        else:
            model_for_test = model_for_test_instance
        
        model_for_test.load_state_dict(checkpoint['model_state_dict'])
        model_for_test = model_for_test.to(device)
        
        print("Best model loaded successfully.")

        final_test_data_loader = Dataset(test_data_raw_orig, time_data_test_orig, shuffle=False, opt=opt_from_checkpoint)
        
        test_hit, test_mrr = evaluate_final_test(model_for_test, final_test_data_loader, opt_from_checkpoint, device)
        
        print("\n--- Final Test Set Evaluation Results ---")
        print(f'Recall@20 on Test Set: {test_hit:.4f}')
        print(f'MRR@20 on Test Set: {test_mrr:.4f}')
        print("="*50)
    else:
        print("\nNo best model (based on recall) was saved, or path is invalid. Skipping final test set evaluation.")


    # --- Plotting ---
    epochs_range = range(len(train_loss_history))

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_loss_history, label='Train Total Loss')
    plt.title('Training Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, train_main_loss_history, label='Train Main Loss')
    plt.title('Training Main Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, train_ssl_loss_history, label='Train SSL Loss')
    plt.title('Training SSL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, val_recall_history, label='Validation Recall@20')
    plt.title('Recall@20')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@20')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, val_mrr_history, label='Validation MRR@20')
    plt.title('MRR@20')
    plt.xlabel('Epoch')
    plt.ylabel('MRR@20')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    ax1 = plt.gca()
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Recall@20', color=color)
    ax1.plot(epochs_range, val_recall_history, color=color, label='Val Recall@20')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Train Total Loss', color=color) 
    ax2.plot(epochs_range, train_loss_history, color=color, linestyle='--', label='Train Total Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    plt.title('Overfitting Check: Val Recall vs. Train Loss')


    plt.tight_layout()
    last_epoch_num = epoch_num if 'epoch_num' in locals() and isinstance(epoch_num, int) else opt.epoch -1
    plot_filename = os.path.join(plot_dir, f'{opt.dataset}_training_plots_epoch_{last_epoch_num}.png')
    plt.savefig(plot_filename)
    print(f"Saved training plots to {plot_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) #
    
    parser.add_argument('--dataset', default='yoochoose1_64', choices=['diginetica', 'yoochoose1_64'], help='Dataset name') #
    parser.add_argument('--validation', type=str2bool, default=True, help='Use validation split') #
    parser.add_argument('--valid_portion', type=float, default=0.1, help='Validation split portion') #

    parser.add_argument('--hiddenSize', type=int, default=512, help='Hidden state dimension') #
    parser.add_argument('--step', type=int, default=4, help='GNN propagation steps') #
    parser.add_argument('--nonhybrid', type=str2bool, default=False, help='Use non-hybrid scoring') #
    parser.add_argument('--max_len', type=int, default=0, help='Max session length (0=auto)') #
    parser.add_argument('--position_emb_dim', type=int, default=0, help='Position embedding dim (0=hiddenSize)') #

    parser.add_argument('--n_gpu', type=int, default=1, help='Num GPUs (0=CPU)') #
    parser.add_argument('--batchSize', type=int, default=512, help='Batch size') #
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs') #
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') #
    parser.add_argument('--l2', type=float, default=1e-4, help='L2 penalty') #
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience') #

    parser.add_argument('--ssl_weight', type=float, default=0.5, help='SSL loss weight') #
    parser.add_argument('--ssl_temperature', type=float, default=0.1, help='SSL temperature') #
    parser.add_argument('--ssl_item_drop_prob', type=float, default=0.4, help='Item dropout prob') #
    parser.add_argument('--ssl_projection_dim', type=int, default=0, help='Projection dim (0=hiddenSize/2)') #

    cmd_args = parser.parse_args() #
    
    opt = argparse.Namespace() #
    
    if cmd_args.dataset == 'diginetica': #
        base_config = Diginetica_arg() #
    elif cmd_args.dataset == 'yoochoose1_64': #
        base_config = Yoochoose_arg() #
    else:
        print(f"Error: Unknown dataset '{cmd_args.dataset}'") #
        sys.exit(1) #
    
    for key, value in vars(base_config).items(): #
        setattr(opt, key, value) #
    
    for key, value in vars(cmd_args).items(): #
        if hasattr(opt, key): #
            setattr(opt, key, value) #
    
    if opt.ssl_projection_dim == 0: #
        opt.ssl_projection_dim = opt.hiddenSize // 2 #

    if opt.position_emb_dim == 0: #
        opt.position_emb_dim = opt.hiddenSize #
        
    print("Final configuration:") #
    for k, v in vars(opt).items(): #
        print(f"{k}: {v}") #
    
    main(opt)
