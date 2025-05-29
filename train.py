import argparse
import pickle
import time
import sys
import os

import torch # <--- اضافه شد
from proc_utils import Dataset, split_validation
from model import * # Imports Attention_SessionGraph, train_test, to_cuda etc.
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    return v.lower() in ('true')

# Default args used for Diginetica
class Diginetica_arg():
    dataset = 'diginetica'
    batchSize = 50 # این بچ‌سایز کلی است که بین GPUها تقسیم می‌شود
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
    batchSize = 75 # این بچ‌سایز کلی است که بین GPUها تقسیم می‌شود
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
    model_save_dir = 'saved_ssl/'
    log_dir = 'with_pos_ssl/logs'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"Directory {model_save_dir} created.")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Directory {log_dir} created.")

    writer = SummaryWriter(log_dir=log_dir)

    # --- تعیین دستگاه و تعداد GPU ---
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpu}")
        if n_gpu < 2 and opt.n_gpu > 1 : # اگر کاربر بیش از ۱ GPU خواسته ولی موجود نیست
            print(f"Warning: Requested {opt.n_gpu} GPUs, but only {n_gpu} are available. Using {n_gpu} GPU(s).")
            opt.n_gpu = n_gpu
        elif n_gpu >= 2 and opt.n_gpu == 0: # اگر کاربر GPU مشخص نکرده ولی موجود است، از همه استفاده کن
             opt.n_gpu = n_gpu # استفاده از همه GPUهای موجود به طور پیش‌فرض
             print(f"Automatically using all {n_gpu} available GPUs.")
        elif opt.n_gpu == 0 and n_gpu < 2: # اگر GPU نیست یا فقط یکی هست
            opt.n_gpu = n_gpu


        if opt.n_gpu > 0:
            device = torch.device("cuda:0") # GPU اصلی برای DataParallel
            print(f"Using {opt.n_gpu} GPU(s). Main device: {device}")
        else:
            device = torch.device("cpu")
            print("Using CPU.")
    else:
        device = torch.device("cpu")
        opt.n_gpu = 0
        print("CUDA not available. Using CPU.")
    # ------------------------------------

    if opt.dataset == 'diginetica':
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
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
        print('Using validation set for testing.')
    else:
        print('Using full test set (no validation split).')

    train_data_loader = Dataset(train_data, shuffle=True)
    test_data_loader = Dataset(test_data, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310
        print(f"Warning: n_node not explicitly set for dataset {opt.dataset}, using fallback {n_node}")

    # --- نمونه‌سازی و انتقال مدل به دستگاه ---
    # ابتدا مدل اصلی را ایجاد می‌کنیم
    model_instance = Attention_SessionGraph(opt, n_node)

    # اگر بیش از یک GPU برای استفاده مشخص شده باشد، از DataParallel استفاده می‌کنیم
    if opt.n_gpu > 1:
        print(f"Using torch.nn.DataParallel for {opt.n_gpu} GPUs.")
        # مشخص کردن device_ids برای DataParallel
        # اگر opt.gpu_ids مشخص نشده باشد، از همه GPUهای موجود تا سقف opt.n_gpu استفاده می‌کند.
        # در Kaggle با ۲ تا T4، معمولا device_ids=[0, 1] خواهد بود.
        gpu_ids_to_use = list(range(opt.n_gpu))
        model = torch.nn.DataParallel(model_instance, device_ids=gpu_ids_to_use)
        model.to(device) # DataParallel مدل را به device_ids[0] منتقل می‌کند و داده‌ها را پخش می‌کند
    elif opt.n_gpu == 1:
        model = model_instance.to(device) # انتقال به GPU تکی
        print(f"Using single GPU: {device}")
    else: # CPU
        model = model_instance
        print("Using CPU for model.")
    # -----------------------------------------

    # `to_cuda` در فایل model.py هم می‌تواند استفاده شود، اما اینجا مدیریت صریح‌تر است.
    # اگر از `to_cuda(model_instance)` استفاده می‌کردید، باید مطمئن می‌شدید که به درستی GPU اصلی را انتخاب می‌کند.

    print(f"Model initialized with n_node={n_node}")
    print(f"Hyperparameters: {vars(opt)}")

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch_num in range(opt.epoch):
        print('-' * 50)
        print('Epoch: ', epoch_num)

        # هنگام فراخوانی train_test، خود مدل (که ممکن است DataParallel باشد) و opt را پاس می‌دهیم
        # تابع train_test در model.py نیازی به دانستن صریح در مورد DataParallel ندارد،
        # چون DataParallel فراخوانی‌های forward و backward را به درستی مدیریت می‌کند.
        # فقط باید مطمئن شویم که داده‌ها به دستگاه صحیح منتقل می‌شوند.
        # تابع to_cuda در model.py اگر دستگاه را به صورت دینامیک انتخاب نکند،
        # ممکن است نیاز به بازبینی داشته باشد، یا اینکه در train_test قبل از ارسال داده به مدل،
        # داده‌ها را به device (که می‌تواند cuda:0 باشد) منتقل کنیم.

        # تغییر در تابع train_test برای مدیریت دستگاه داده‌ها:
        # در فایل model.py، تابع to_cuda(tensor) باید به نحوی تغییر کند که
        # tensor.to(model.device) یا tensor.to(next(model.parameters()).device) را انجام دهد
        # تا داده‌ها به همان دستگاهی بروند که مدل روی آن است (مخصوصا GPU اصلی برای DataParallel).
        # یا اینکه `device` را به `train_test` پاس دهیم. راه دوم ساده‌تر است.

        hit, mrr = train_test(model, train_data_loader, test_data_loader, opt, device) # <--- device اضافه شد

        print('Epoch %d finished. Recall@20: %.4f, MRR@20: %.4f' % (epoch_num, hit, mrr))

        writer.add_scalar('epoch/recall_at_20', hit, epoch_num)
        writer.add_scalar('epoch/mrr_at_20', mrr, epoch_num)

        flag = 0
        current_recall_save_path = "" # برای جلوگیری از خطای متغیر تعریف نشده

        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch_num
            flag = 1
            current_recall_save_path = model_save_dir + f'epoch_{epoch_num}_recall_{hit:.4f}.pt'
            # برای ذخیره مدل، اگر از DataParallel استفاده شده، state_dict از model.module گرفته می‌شود
            state_to_save = model.module.state_dict() if opt.n_gpu > 1 else model.state_dict()
            torch.save(state_to_save, current_recall_save_path)
            print(f"Saved best recall model to {current_recall_save_path}")

        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch_num
            flag = 1
            current_mrr_save_path = model_save_dir + f'epoch_{epoch_num}_mrr_{mrr:.4f}.pt'
            if not (flag == 1 and hit >= best_result[0] and current_mrr_save_path == current_recall_save_path):
                state_to_save = model.module.state_dict() if opt.n_gpu > 1 else model.state_dict()
                torch.save(state_to_save, current_mrr_save_path)
                print(f"Saved best MRR model to {current_mrr_save_path}")

        print('Current Best Result:')
        print('\tRecall@20: %.4f (Epoch %d)\tMRR@20: %.4f (Epoch %d)' %
              (best_result[0], best_epoch[0], best_result[1], best_epoch[1]))

        bad_counter += (1 - flag)
        if bad_counter >= opt.patience:
            print(f"Early stopping triggered after {opt.patience} epochs without improvement.")
            break

    writer.close()
    print('-' * 50)
    end = time.time()
    print("Total Running time: %f seconds" % (end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', help='Dataset name: diginetica | yoochoose1_64')
    parser.add_argument('--defaults', type=str2bool, default=True, help='Use default configuration for the chosen dataset (True/False)')
    parser.add_argument('--n_gpu', type=int, default=0, help='Number of GPUs to use (0 for CPU, if available uses all, or up to 2 for Kaggle T4s)') # <--- آرگومان جدید

    # General Hyperparameters
    parser.add_argument('--batchSize', type=int, default=0, help='Batch size (0 uses dataset default)') # پیش‌فرض ۰ برای استفاده از تنظیمات کلاس
    parser.add_argument('--hiddenSize', type=int, default=0, help='Hidden state dimensions (0 uses dataset default)')
    parser.add_argument('--epoch', type=int, default=0, help='The number of epochs to train for (0 uses dataset default)')
    # ... (بقیه آرگومان‌ها مانند قبل) ...
    parser.add_argument('--lr', type=float, default=0, help='Learning Rate (0 uses dataset default)')
    parser.add_argument('--lr_dc', type=float, default=0, help='Learning rate decay rate (0 uses dataset default)')
    parser.add_argument('--lr_dc_step', type=int, default=0, help='Steps for learning rate decay (0 uses dataset default)')
    parser.add_argument('--l2', type=float, default=0, help='L2 Penalty (0 uses dataset default)')
    parser.add_argument('--step', type=int, default=0, help='GNN propagation steps (0 uses dataset default)')
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 uses dataset default)')
    parser.add_argument('--nonhybrid', type=str2bool, default=None, help='Whether to use non-hybrid model (None uses dataset default)')
    parser.add_argument('--validation', type=str2bool, default=None, help='Whether to use a validation split (None uses dataset default)')
    parser.add_argument('--valid_portion', type=float, default=0, help='Portion of train set for validation (0 uses dataset default)')

    # SSL Specific Hyperparameters
    parser.add_argument('--ssl_weight', type=float, default=-1.0, help='Weight for SSL loss component (-1.0 uses dataset default)')
    parser.add_argument('--ssl_temperature', type=float, default=-1.0, help='Temperature for InfoNCE SSL loss (-1.0 uses dataset default)')
    parser.add_argument('--ssl_item_drop_prob', type=float, default=-1.0, help='Item drop probability for SSL augmentation (-1.0 uses dataset default)')
    parser.add_argument('--ssl_projection_dim', type=int, default=-1, help='Dimension of SSL projection head output (-1 uses dataset default, 0 for hiddenSize/2)')


    cmd_opt = parser.parse_args()
    temp_opt = None # برای نگهداری تنظیمات پیش‌فرض دیتاست

    if cmd_opt.dataset == 'diginetica':
        temp_opt = Diginetica_arg()
    elif cmd_opt.dataset == 'yoochoose1_64':
        temp_opt = Yoochoose_arg()
    else:
        print(f"Error: Unknown dataset '{cmd_opt.dataset}' for default configurations.")
        sys.exit(1)

    # اگر کاربر defaults=True را مشخص کرده، ابتدا از کلاس‌های پیش‌فرض استفاده کن
    # سپس مقادیری که کاربر صراحتا در خط فرمان وارد کرده را جایگزین کن
    if cmd_opt.defaults:
        opt = temp_opt
        # Override with any explicitly passed command-line args
        for key, value in vars(cmd_opt).items():
            # فقط اگر مقدار در خط فرمان با مقدار پیش‌فرض آرگومان متفاوت بود، آن را جایگزین کن
            # و همچنین اگر آن کلید در کلاس پیش‌فرض دیتاست وجود داشت
            if hasattr(opt, key) and value != parser.get_default(key):
                print(f"Overriding default '{key}' from '{getattr(opt, key)}' with command-line value: {value}")
                setattr(opt, key, value)
    else: # اگر defaults=False بود، فقط از مقادیر خط فرمان استفاده کن
        print("Not using default dataset configurations. Using command-line arguments directly.")
        opt = cmd_opt # همه مقادیر از خط فرمان می‌آیند
        # مقادیری که کاربر وارد نکرده و پیش‌فرض آرگومان هستند (مثل ۰ یا None) باید مدیریت شوند
        # و با مقادیر معقول از temp_opt (که کلاس پیش‌فرض دیتاست است) پر شوند
        for key, value in vars(cmd_opt).items():
            default_arg_val = parser.get_default(key)
            if value == default_arg_val and hasattr(temp_opt, key): # اگر کاربر مقدار پیش‌فرض آرگومان را استفاده کرده
                print(f"Using dataset default for '{key}': {getattr(temp_opt, key)}")
                setattr(opt, key, getattr(temp_opt, key))


    # تنظیم ssl_projection_dim اگر کاربر مقدار خاصی نداده باشد
    if opt.ssl_projection_dim == -1 or (opt.ssl_projection_dim == 0 and not cmd_opt.defaults and cmd_opt.ssl_projection_dim == 0): # حالت پیش‌فرض یا 0 که یعنی محاسبه شود
        opt.ssl_projection_dim = opt.hiddenSize // 2
        print(f"Calculated ssl_projection_dim: {opt.ssl_projection_dim}")
    elif opt.ssl_projection_dim == 0 and (cmd_opt.defaults or (not cmd_opt.defaults and cmd_opt.ssl_projection_dim != 0)): # اگر کاربر صراحتا ۰ وارد کرده
        print(f"Warning: ssl_projection_dim is explicitly set to 0. Using hiddenSize // 2 instead.")
        opt.ssl_projection_dim = opt.hiddenSize // 2


    main(opt)
