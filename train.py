import argparse
import pickle
import time
import sys
import os # <--- ماژول os اضافه شد

from proc_utils import Dataset, split_validation
from model import * # Imports Attention_SessionGraph, train_test, to_cuda etc.
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    return v.lower() in ('true')

# Default args used for Diginetica
class Diginetica_arg():
    dataset = 'diginetica'
    batchSize = 50
    hiddenSize = 100
    epoch = 30
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1 # GNN step
    patience = 10
    nonhybrid = True # For model.compute_scores
    validation = True
    valid_portion = 0.1
    # SSL Args
    ssl_weight = 0.1
    ssl_temperature = 0.07
    ssl_item_drop_prob = 0.2
    ssl_projection_dim = 50 # Example, hiddenSize // 2


# Default args used for Yoochoose1_64
class Yoochoose_arg():
    dataset = 'yoochoose1_64'
    batchSize = 75 # Original: 75
    hiddenSize = 120 # Original: 120
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
    # SSL Args
    ssl_weight = 0.1
    ssl_temperature = 0.07
    ssl_item_drop_prob = 0.2
    ssl_projection_dim = 60 # Example, hiddenSize // 2


def main(opt):
    model_save_dir = 'saved_ssl/' # Changed save directory
    log_dir = 'with_pos_ssl/logs' # Changed log directory

    # --- ایجاد پوشه برای ذخیره مدل و لاگ‌ها در صورت عدم وجود ---
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"Directory {model_save_dir} created.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Directory {log_dir} created.")
    # ---------------------------------------------------------

    writer = SummaryWriter(log_dir=log_dir)

    if opt.dataset == 'diginetica':
        # Ensure dataset paths are correct
        train_data_path = 'datasets/cikm16/raw/train.txt'
        test_data_path = 'datasets/cikm16/raw/test.txt'
        try:
            train_data = pickle.load(open(train_data_path, 'rb'))
            test_data = pickle.load(open(test_data_path, 'rb'))
        except FileNotFoundError:
            print(f"Error: Dataset file not found. Searched at {train_data_path} and {test_data_path}")
            print("Please ensure datasets are correctly placed.")
            sys.exit(1)


    elif opt.dataset == 'yoochoose1_64':
        train_data_path = 'datasets/yoochoose1_64/raw/train.txt'
        test_data_path = 'datasets/yoochoose1_64/raw/test.txt'
        try:
            train_data = pickle.load(open(train_data_path, 'rb'))
            test_data = pickle.load(open(test_data_path, 'rb'))
        except FileNotFoundError:
            print(f"Error: Dataset file not found. Searched at {train_data_path} and {test_data_path}")
            print("Please ensure datasets are correctly placed.")
            sys.exit(1)
    else:
        print(f"Error: Unknown dataset {opt.dataset}")
        sys.exit(1)


    if opt.validation:
        train_data, valid_data = split_validation(
            train_data, opt.valid_portion)
        test_data = valid_data # Evaluate on validation set if validation is True
        print('Using validation set for testing.')
    else:
        print('Using full test set (no validation split).')


    train_data_loader = Dataset(train_data, shuffle=True)
    test_data_loader = Dataset(test_data, shuffle=False)


    if opt.dataset == 'diginetica':
        n_node = 43098 # Number of items + 1 for padding (if 0 is padding)
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4': # yoochoose1_4 not handled for path
        n_node = 37484
    else: # Fallback, should be defined per dataset
        n_node = 310
        print(f"Warning: n_node not explicitly set for dataset {opt.dataset}, using fallback {n_node}")

    model = to_cuda(Attention_SessionGraph(opt, n_node)) # opt is passed to model for SSL params

    print(f"Model initialized with n_node={n_node}")
    print(f"Hyperparameters: {vars(opt)}")

    start = time.time()
    best_result = [0, 0] # [hit, mrr]
    best_epoch = [0, 0]  # [epoch_for_hit, epoch_for_mrr]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-' * 50)
        print('Epoch: ', epoch)
        # Pass opt to train_test
        hit, mrr = train_test(model, train_data_loader, test_data_loader, opt)

        print('Epoch %d finished. Recall@20: %.4f, MRR@20: %.4f' % (epoch, hit, mrr))

        # Logging to TensorBoard
        writer.add_scalar('epoch/recall_at_20', hit, epoch)
        writer.add_scalar('epoch/mrr_at_20', mrr, epoch)
        # Can also log average losses from train_test if returned

        flag = 0 # Flag to check if this epoch was the best for any metric

        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            # Save model if it's the best for recall
            save_path = model_save_dir + f'epoch_{epoch}_recall_{hit:.4f}.pt'
            torch.save(model.state_dict(), save_path) # <--- قبلاً اینجا model ذخیره می‌شد، الان state_dict ذخیره می‌شود
            print(f"Saved best recall model to {save_path}")

        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            # Save model if it's the best for MRR (could be same as recall or different)
            # فقط در صورتی ذخیره کن که نام فایل متفاوت باشد (برای جلوگیری از بازنویسی در صورتی که هر دو معیار همزمان بهترین شوند)
            current_mrr_save_path = model_save_dir + f'epoch_{epoch}_mrr_{mrr:.4f}.pt'
            if not (flag == 1 and hit >= best_result[0] and current_mrr_save_path == save_path): # اگر برای recall ذخیره شده و نام یکی است، دوباره ذخیره نکن
                 torch.save(model.state_dict(), current_mrr_save_path) # <--- قبلاً اینجا model ذخیره می‌شد، الان state_dict ذخیره می‌شود
                 print(f"Saved best MRR model to {current_mrr_save_path}")


        print('Current Best Result:')
        print('\tRecall@20: %.4f (Epoch %d)\tMRR@20: %.4f (Epoch %d)' %
              (best_result[0], best_epoch[0], best_result[1], best_epoch[1]))

        bad_counter += (1 - flag) # Increment if neither metric improved

        if bad_counter >= opt.patience:
            print(f"Early stopping triggered after {opt.patience} epochs without improvement.")
            break

    writer.close()
    print('-' * 50)
    end = time.time()
    print("Total Running time: %f seconds" % (end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica',
                        help='Dataset name: diginetica | yoochoose1_64')
    parser.add_argument('--defaults', type=str2bool, default=True,
                        help='Use default configuration for the chosen dataset (True/False)')

    # General Hyperparameters (can be overridden if defaults is False)
    parser.add_argument('--batchSize', type=int, default=50, help='Batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='Hidden state dimensions')
    parser.add_argument('--epoch', type=int, default=30, help='The number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='Steps for learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 Penalty')
    parser.add_argument('--step', type=int, default=1, help='GNN propagation steps')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--nonhybrid', type=str2bool, default=True, help='Whether to use non-hybrid model for final score calculation')
    parser.add_argument('--validation', type=str2bool, default=True, help='Whether to use a validation split') # Changed from action='store_true'
    parser.add_argument('--valid_portion', type=float, default=0.1, help='Portion of train set for validation')

    # SSL Specific Hyperparameters
    parser.add_argument('--ssl_weight', type=float, default=0.1, help='Weight for SSL loss component')
    parser.add_argument('--ssl_temperature', type=float, default=0.07, help='Temperature for InfoNCE SSL loss')
    parser.add_argument('--ssl_item_drop_prob', type=float, default=0.2, help='Item drop probability for SSL augmentation')
    parser.add_argument('--ssl_projection_dim', type=int, default=0, help='Dimension of SSL projection head output (0 for hiddenSize/2)')


    cmd_opt = parser.parse_args()

    if cmd_opt.defaults:
        if cmd_opt.dataset == 'diginetica':
            opt = Diginetica_arg()
        elif cmd_opt.dataset == 'yoochoose1_64':
            opt = Yoochoose_arg()
        else:
            print(f"Warning: Default arguments not defined for dataset '{cmd_opt.dataset}'. Using command line or general defaults.")
            # Fallback to command line args if dataset specific defaults are not found but defaults=True
            opt = cmd_opt
            # Manually set projection dim if it's 0 from cmd_opt and needs calculation
            if hasattr(opt, 'ssl_projection_dim') and opt.ssl_projection_dim == 0:
                 opt.ssl_projection_dim = opt.hiddenSize // 2

        # Override default class args with any explicitly passed command-line args
        # This allows using defaults but tweaking a few params.
        for key, value in vars(cmd_opt).items():
            # Only override if CMD arg is not its own default value provided in add_argument
            if hasattr(opt, key) and value != parser.get_default(key):
                print(f"Overriding default '{key}' with command-line value: {value}")
                setattr(opt, key, value)

        # Ensure ssl_projection_dim is set if it was default 0 from class
        if hasattr(opt, 'ssl_projection_dim') and opt.ssl_projection_dim == 0 and hasattr(opt, 'hiddenSize'):
            opt.ssl_projection_dim = opt.hiddenSize // 2


    else:
        print("Not using default dataset configurations. Using command-line arguments directly.")
        opt = cmd_opt
        if opt.ssl_projection_dim == 0: # If not set by user and not using defaults
            opt.ssl_projection_dim = opt.hiddenSize // 2
            print(f"Setting ssl_projection_dim to hiddenSize/2 = {opt.ssl_projection_dim}")


    main(opt)
