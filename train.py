import argparse
import pickle
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Ditambahkan untuk run_evaluation

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

def get_n_node_from_data(train_data_raw, test_data_raw_orig):
    max_item_id = 0
    for seq_list in [train_data_raw[0], test_data_raw_orig[0]]:
        for seq in seq_list:
            for item_id in seq:
                if item_id > max_item_id:
                    max_item_id = item_id
    return max_item_id + 1 # +1 for padding index 0

def run_evaluation(model_to_eval, data_loader, opt, device, eval_desc="Evaluation"):
    # Pastikan model_to_eval adalah instance model, bukan DataParallel wrapper jika perlu
    model_module = model_to_eval.module if isinstance(model_to_eval, torch.nn.DataParallel) else model_to_eval
    model_to_eval.eval() # Set model ke mode evaluasi
    hit, mrr = [], []
    slices_eval = data_loader.generate_batch(opt.batchSize)

    if not slices_eval:
        print(f"Peringatan: Tidak ada batch yang dihasilkan dari data {eval_desc}. Melewati evaluasi.")
        return 0.0, 0.0

    with torch.no_grad(): # Tidak perlu menghitung gradien saat evaluasi
        for i_eval_slice_indices in tqdm(slices_eval, desc=f"{eval_desc}"):
            if len(i_eval_slice_indices) == 0:
                continue

            # Menggunakan _v1 karena SSL tidak aktif selama evaluasi murni
            # Struktur data yang dikembalikan oleh get_slice adalah:
            # (alias, A, items_unique, mask_ssl, positions), _, targets_np, mask_main_np, time_diffs_v1, _
            data_v1_eval_tuple, _, targets_eval_np, mask_eval_np, time_diffs_eval_np, _ = data_loader.get_slice(
                i_eval_slice_indices, ssl_item_drop_prob=0.0 # Tidak ada item drop saat evaluasi
            )

            alias_inputs_eval, A_eval, items_eval_unique, _, position_ids_eval = data_v1_eval_tuple

            if items_eval_unique.size == 0:
                print(f"Melewati batch {eval_desc} karena array item unik kosong.")
                continue

            items_eval_unique = torch.from_numpy(items_eval_unique).long().to(device)
            A_eval = torch.from_numpy(A_eval).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval).long().to(device)
            time_diffs_eval = torch.from_numpy(time_diffs_eval_np).float().to(device)

            mask_eval_cuda = torch.from_numpy(mask_eval_np).long().to(device)
            targets_eval_cuda = torch.from_numpy(targets_eval_np).long().to(device)

            # Panggil model. `None` untuk sequence_mask_for_attention yang sudah dihapus
            final_seq_hidden_eval, star_context_eval = model_to_eval(items_eval_unique, A_eval, alias_inputs_eval, position_ids_eval, time_diffs_eval, None)

            if final_seq_hidden_eval.size(0) == 0:
                print(f"Melewati batch {eval_desc} karena final_seq_hidden_eval kosong.")
                continue

            scores_eval = model_module.compute_scores(final_seq_hidden_eval, star_context_eval, mask_eval_cuda)

            sub_scores_top20_indices = scores_eval.topk(20)[1]
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()

            # Target dikurangi 1 karena embedding dimulai dari 1, tapi target array 0-indexed
            targets_eval_for_metric_np = targets_eval_cuda.cpu().detach().numpy() - 1

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_for_metric_np):
                hit.append(np.isin(target_item, score_row))
                if target_item in score_row:
                    rank = np.where(score_row == target_item)[0][0] + 1
                    mrr.append(1.0 / rank)
                else:
                    mrr.append(0.0)

    eval_hit_metric = np.mean(hit) * 100 if hit else 0.0
    eval_mrr_metric = np.mean(mrr) * 100 if mrr else 0.0

    return eval_hit_metric, eval_mrr_metric


def main(opt):
    data_dir = f'datasets/{opt.dataset}/'
    model_save_dir = 'saved_star_models/'
    log_dir = 'logs_star/'
    plot_dir = 'plots_star/'

    for directory in [model_save_dir, log_dir, plot_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Direktori {directory} dibuat.")

    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.n_gpu > 0 else "cpu")
    if torch.cuda.is_available() and opt.n_gpu > 0:
        print(f"Menggunakan {torch.cuda.device_count()} GPU(s)")
    else:
        print("Menggunakan CPU")

    try:
        with open(os.path.join(data_dir, 'train.txt'), 'rb') as f:
            train_data_raw = pickle.load(f)
        with open(os.path.join(data_dir, 'test.txt'), 'rb') as f:
            test_data_raw_orig = pickle.load(f) # Data tes asli
        with open(os.path.join(data_dir, 'time_data.pkl'), 'rb') as f:
            time_data_all = pickle.load(f)
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        sys.exit(1)

    time_data_train_orig = time_data_all['train_time_diffs']
    time_data_test_orig = time_data_all['test_time_diffs'] # Time data untuk tes asli

    n_node = get_n_node_from_data(train_data_raw, test_data_raw_orig)
    print(f"n_node yang dihitung secara dinamis: {n_node}")

    train_loss_history = []
    train_main_loss_history = []
    train_ssl_loss_history = []
    val_recall_history = []
    val_mrr_history = []

    if opt.validation:
        (train_seqs, train_targets), (valid_seqs, valid_targets) = split_validation(
            train_data_raw, opt.valid_portion
        )
        train_data_raw = (train_seqs, train_targets)
        test_data_raw_for_eval = (valid_seqs, valid_targets) # Untuk validasi epoch
        
        n_samples = len(time_data_train_orig)
        indices = list(range(n_samples))
        np.random.shuffle(indices)
        n_train = int(np.round(n_samples * (1. - opt.valid_portion)))
        
        time_data_train = [time_data_train_orig[i] for i in indices[:n_train]]
        time_data_valid_for_eval = [time_data_train_orig[i] for i in indices[n_train:]]
    else:
        test_data_raw_for_eval = test_data_raw_orig # Jika tidak ada validasi, evaluasi epoch pakai data tes asli
        time_data_train = time_data_train_orig
        time_data_valid_for_eval = time_data_test_orig

    train_data_loader = Dataset(train_data_raw, time_data_train, shuffle=True, opt=opt)
    eval_data_loader = Dataset(test_data_raw_for_eval, time_data_valid_for_eval, shuffle=False, opt=opt)
    
    actual_dataset_max_len = train_data_loader.len_max
    if opt.max_len == 0 or opt.max_len < actual_dataset_max_len:
        print(f"Memperbarui opt.max_len dari {opt.max_len} ke {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    if opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
        
    if opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2

    model_instance = Attention_SessionGraph(opt, n_node)

    if opt.n_gpu > 1 and torch.cuda.is_available():
        print(f"Menggunakan {opt.n_gpu} GPU dengan DataParallel")
        model = torch.nn.DataParallel(model_instance)
    else:
        model = model_instance
        
    model = model.to(device)

    print(f"Model diinisialisasi. n_node={n_node}, max_session_len={opt.max_len}, pos_emb_dim={opt.position_emb_dim}")
    print(f"Hyperparameters: {vars(opt)}")

    start_time = time.time()
    best_result = [0.0, 0.0] 
    best_epoch = [0, 0]
    bad_counter = 0
    best_model_path_recall = None # Path untuk model dengan recall terbaik

    for epoch_num in range(opt.epoch):
        print('-' * 50 + f'\nEpoch: {epoch_num}')
        opt.current_epoch_num = epoch_num
        
        eval_hit, eval_mrr, avg_train_total_loss, avg_train_main_loss, avg_train_ssl_loss = train_test(
            model, train_data_loader, eval_data_loader, opt, device
        )

        print(f'Epoch {epoch_num} Eval - Recall@20: {eval_hit:.4f}, MRR@20: {eval_mrr:.4f}')
        print(f'Epoch {epoch_num} Train Avg Loss - Total: {avg_train_total_loss:.4f}, Main: {avg_train_main_loss:.4f}, SSL: {avg_train_ssl_loss:.4f}')

        writer.add_scalar('epoch/eval_recall_at_20', eval_hit, epoch_num)
        writer.add_scalar('epoch/eval_mrr_at_20', eval_mrr, epoch_num)
        writer.add_scalar('epoch/train_total_loss', avg_train_total_loss, epoch_num)
        writer.add_scalar('epoch/train_main_loss', avg_train_main_loss, epoch_num)
        writer.add_scalar('epoch/train_ssl_loss', avg_train_ssl_loss, epoch_num)
        
        train_loss_history.append(avg_train_total_loss)
        train_main_loss_history.append(avg_train_main_loss)
        train_ssl_loss_history.append(avg_train_ssl_loss)
        val_recall_history.append(eval_hit)
        val_mrr_history.append(eval_mrr)

        flag = 0
        saved_this_epoch_path = ""

        if eval_hit >= best_result[0]:
            best_result[0] = eval_hit
            best_epoch[0] = epoch_num
            flag = 1
            # Simpan path model terbaik berdasarkan recall
            best_model_path_recall = os.path.join(model_save_dir, f'best_recall_epoch_{epoch_num}_recall_{eval_hit:.4f}_mrr_{eval_mrr:.4f}.pt')
            state_to_save = {
                'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'opt': opt,
                'epoch': epoch_num,
                'recall': eval_hit,
                'mrr': eval_mrr
            }
            torch.save(state_to_save, best_model_path_recall)
            print(f"Menyimpan model terbaik (berdasarkan Recall) ke {best_model_path_recall}")

        if eval_mrr >= best_result[1]:
            best_result[1] = eval_mrr
            best_epoch[1] = epoch_num
            flag = 1
            mrr_save_path = os.path.join(model_save_dir, f'best_mrr_epoch_{epoch_num}_recall_{eval_hit:.4f}_mrr_{eval_mrr:.4f}.pt')
            if mrr_save_path != best_model_path_recall: # Hanya simpan jika path berbeda (atau jika ini adalah peningkatan MRR pertama)
                state_to_save = {
                    'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'opt': opt,
                    'epoch': epoch_num,
                    'recall': eval_hit,
                    'mrr': eval_mrr
                }
                torch.save(state_to_save, mrr_save_path)
                print(f"Menyimpan model terbaik (berdasarkan MRR) ke {mrr_save_path}")

        print(f'Terbaik Saat Ini: Recall@20: {best_result[0]:.4f} (Epoch {best_epoch[0]}), MRR@20: {best_result[1]:.4f} (Epoch {best_epoch[1]})')
        
        bad_counter = 0 if flag else bad_counter + 1
        if bad_counter >= opt.patience:
            print(f"Early stopping setelah {opt.patience} epoch tanpa peningkatan.")
            break
            
    writer.close()
    print('-' * 50 + f"\nTotal Waktu Berjalan: {(time.time() - start_time)/3600:.2f} jam")

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
    plot_filename = os.path.join(plot_dir, f'{opt.dataset}_training_plots_epoch_{epoch_num}.png')
    plt.savefig(plot_filename)
    print(f"Plot training disimpan ke {plot_filename}")
    # plt.show() # Komentari jika berjalan di lingkungan non-GUI

    # --- Evaluasi Akhir pada Test Set Asli ---
    print('-' * 50)
    print("Memulai evaluasi akhir pada dataset tes asli...")

    if best_model_path_recall and os.path.exists(best_model_path_recall):
        print(f"Memuat model terbaik untuk evaluasi tes akhir dari: {best_model_path_recall}")
        
        # Inisialisasi ulang struktur model
        final_test_model_instance = Attention_SessionGraph(opt, n_node) # opt dan n_node dari training
        if opt.n_gpu > 1 and torch.cuda.is_available():
             final_test_model = torch.nn.DataParallel(final_test_model_instance)
        else:
            final_test_model = final_test_model_instance
        final_test_model = final_test_model.to(device)

        # Muat state yang disimpan
        checkpoint = torch.load(best_model_path_recall, map_location=device)
        
        model_state_dict = checkpoint['model']
        # Penanganan jika model disimpan dengan atau tanpa DataParallel
        is_model_parallel = isinstance(final_test_model, torch.nn.DataParallel)
        is_checkpoint_parallel = list(model_state_dict.keys())[0].startswith('module.')

        if is_model_parallel and not is_checkpoint_parallel:
            # Model saat ini DataParallel, tapi checkpoint tidak. Tambahkan 'module.'
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}
        elif not is_model_parallel and is_checkpoint_parallel:
            # Model saat ini bukan DataParallel, tapi checkpoint iya. Hapus 'module.'
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        final_test_model.load_state_dict(model_state_dict)
        print("Model terbaik berhasil dimuat.")

        # Buat DataLoader untuk dataset tes asli
        print("Menyiapkan data tes asli untuk evaluasi akhir...")
        # test_data_raw_orig dan time_data_test_orig sudah dimuat di awal
        final_test_data_loader = Dataset(test_data_raw_orig, time_data_test_orig, shuffle=False, opt=opt)
        
        # Jalankan evaluasi
        test_hit_final, test_mrr_final = run_evaluation(
            final_test_model, final_test_data_loader, opt, device, eval_desc="Evaluasi Set Tes Akhir"
        )
        
        print(f'Hasil Evaluasi Set Tes Akhir:')
        print(f'Recall@20: {test_hit_final:.4f}')
        print(f'MRR@20: {test_mrr_final:.4f}')
        
        # Simpan hasil tes akhir ke file
        results_summary_path = os.path.join(plot_dir, f'{opt.dataset}_final_test_results_epoch_{checkpoint["epoch"]}.txt')
        with open(results_summary_path, 'w') as f_res:
            f_res.write(f"Evaluasi akhir pada set tes menggunakan model: {best_model_path_recall}\n")
            f_res.write(f"Epoch model: {checkpoint['epoch']}\n")
            f_res.write(f"Recall@20 (Validation Saat Simpan): {checkpoint['recall']:.4f}\n")
            f_res.write(f"MRR@20 (Validation Saat Simpan): {checkpoint['mrr']:.4f}\n")
            f_res.write(f"Recall@20 (Test Akhir): {test_hit_final:.4f}\n")
            f_res.write(f"MRR@20 (Test Akhir): {test_mrr_final:.4f}\n")
        print(f"Hasil tes akhir disimpan ke {results_summary_path}")

    else:
        print("Tidak ada model terbaik (berdasarkan recall) yang disimpan selama training, atau path tidak valid. Melewati evaluasi tes akhir.")
    print('-' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', default='yoochoose1_64', choices=['diginetica', 'yoochoose1_64'], help='Nama dataset')
    parser.add_argument('--validation', type=str2bool, default=True, help='Gunakan pemisahan validasi')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='Porsi pemisahan validasi')

    parser.add_argument('--hiddenSize', type=int, default=512, help='Dimensi hidden state')
    parser.add_argument('--step', type=int, default=4, help='Langkah propagasi GNN')
    parser.add_argument('--nonhybrid', type=str2bool, default=False, help='Gunakan scoring non-hybrid')
    parser.add_argument('--max_len', type=int, default=0, help='Panjang sesi maks (0=otomatis)')
    parser.add_argument('--position_emb_dim', type=int, default=0, help='Dimensi embedding posisi (0=hiddenSize)')

    parser.add_argument('--n_gpu', type=int, default=1, help='Jumlah GPU (0=CPU)')
    parser.add_argument('--batchSize', type=int, default=512, help='Ukuran batch')
    parser.add_argument('--epoch', type=int, default=100, help='Jumlah epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='Penalti L2')
    parser.add_argument('--patience', type=int, default=20, help='Kesabaran early stopping')

    parser.add_argument('--ssl_weight', type=float, default=0.5, help='Bobot loss SSL')
    parser.add_argument('--ssl_temperature', type=float, default=0.1, help='Temperatur SSL')
    parser.add_argument('--ssl_item_drop_prob', type=float, default=0.4, help='Probabilitas item dropout SSL')
    parser.add_argument('--ssl_projection_dim', type=int, default=0, help='Dimensi proyeksi SSL (0=hiddenSize/2)')

    cmd_args = parser.parse_args()
    
    opt = argparse.Namespace()
    
    if cmd_args.dataset == 'diginetica':
        base_config = Diginetica_arg()
    elif cmd_args.dataset == 'yoochoose1_64':
        base_config = Yoochoose_arg()
    else:
        print(f"Error: Dataset tidak dikenal '{cmd_args.dataset}'")
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
        
    print("Konfigurasi akhir:")
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    
    main(opt)