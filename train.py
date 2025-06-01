import argparse
import pickle
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from proc_utils import Dataset, split_validation # فایل ابزارهای پردازش داده
from model import Attention_SessionGraph, train_test # فایل مدل و تابع آموزش/تست
from torch.utils.tensorboard import SummaryWriter # برای لاگ کردن در TensorBoard

def str2bool(v):
    # تابعی برای تبدیل رشته به بولین برای آرگومان‌های ورودی
    return v.lower() in ('true', '1', 'yes') #

# کلاس‌های تنظیمات پیش‌فرض برای دیتاست‌های مختلف
class Diginetica_arg:
    def __init__(self):
        self.dataset = 'diginetica' # نام دیتاست
        self.batchSize = 256 # اندازه بچ
        self.hiddenSize = 512 # اندازه لایه پنهان
        self.epoch = 100 # تعداد اپوک‌ها
        self.lr = 0.001 # نرخ یادگیری
        self.l2 = 1e-4 # ضریب L2 regularization
        self.step = 3 # تعداد گام‌های انتشار در GNN
        self.patience = 20 # تعداد اپوک برای early stopping
        self.nonhybrid = False # استفاده از روش امتیازدهی غیرهیبریدی
        self.validation = True # استفاده از تقسیم ولیدیشن
        self.valid_portion = 0.1 # نسبت داده ولیدیشن
        self.ssl_weight = 0.5 # وزن لاس SSL
        self.ssl_temperature = 0.1 # دمای SSL
        self.ssl_item_drop_prob = 0.4 # احتمال حذف آیتم در SSL
        self.ssl_projection_dim = 256 # ابعاد لایه پروجکشن SSL
        self.n_gpu = 1 # تعداد GPU
        self.max_len = 0 # حداکثر طول سکانس (0 به معنی خودکار)
        self.position_emb_dim = 0 # ابعاد امبدینگ موقعیت (0 به معنی hiddenSize)

class Yoochoose_arg:
    def __init__(self):
        self.dataset = 'yoochoose1_64' # نام دیتاست
        self.batchSize = 512 # اندازه بچ
        self.hiddenSize = 512 # اندازه لایه پنهان
        self.epoch = 100 # تعداد اپوک‌ها
        self.lr = 0.001 # نرخ یادگیری
        self.l2 = 1e-4 # ضریب L2 regularization
        self.step = 4 # تعداد گام‌های انتشار در GNN
        self.patience = 20 # تعداد اپوک برای early stopping
        self.nonhybrid = False # استفاده از روش امتیازدهی غیرهیبریدی
        self.validation = True # استفاده از تقسیم ولیدیشن
        self.valid_portion = 0.1 # نسبت داده ولیدیشن
        self.ssl_weight = 0.5 # وزن لاس SSL
        self.ssl_temperature = 0.1 # دمای SSL
        self.ssl_item_drop_prob = 0.4 # احتمال حذف آیتم در SSL
        self.ssl_projection_dim = 256 # ابعاد لایه پروجکشن SSL
        self.n_gpu = 1 # تعداد GPU
        self.max_len = 0 # حداکثر طول سکانس (0 به معنی خودکار)
        self.position_emb_dim = 0 # ابعاد امبدینگ موقعیت (0 به معنی hiddenSize)

def get_n_node_from_data(train_data_raw, test_data_raw_orig):
    # تابعی برای محاسبه تعداد کل آیتم‌های یکتا (n_node) از داده‌های آموزشی و تست
    max_item_id = 0
    # بررسی داده‌های آموزشی
    if train_data_raw and train_data_raw[0]: # train_data_raw[0] شامل لیست سکانس‌ها است
        for seq_list in [train_data_raw[0]]:
            for seq in seq_list:
                for item_id in seq:
                    if item_id > max_item_id:
                        max_item_id = item_id
    # بررسی داده‌های تست اصلی
    if test_data_raw_orig and test_data_raw_orig[0]: # test_data_raw_orig[0] شامل لیست سکانس‌ها است
        for seq_list in [test_data_raw_orig[0]]:
            for seq in seq_list:
                for item_id in seq:
                    if item_id > max_item_id:
                        max_item_id = item_id
    return max_item_id + 1 # +1 به دلیل اینکه ID آیتم‌ها از 1 شروع شده و 0 برای پدینگ است

# تابع جدید برای ارزیابی روی دیتاست تست نهایی
def evaluate_on_test_set(model_path, test_data_loader, opt, device, n_node):
    print(f"\n--- ارزیابی نهایی روی مجموعه داده تست ---")
    print(f"در حال بارگذاری مدل از: {model_path}")

    # ایجاد یک نمونه جدید از مدل برای ارزیابی
    model_instance_eval = Attention_SessionGraph(opt, n_node)
    if opt.n_gpu > 1 and torch.cuda.is_available():
        model_eval = torch.nn.DataParallel(model_instance_eval)
    else:
        model_eval = model_instance_eval
    
    # بررسی وجود فایل مدل قبل از بارگذاری
    if not os.path.exists(model_path):
        print(f"خطا: فایل مدل در مسیر {model_path} یافت نشد.")
        print(f"ارزیابی نهایی روی مجموعه تست انجام نمی‌شود.")
        return 0.0, 0.0 # برگرداندن مقادیر پیش‌فرض

    # بارگذاری checkpoint مدل ذخیره شده
    checkpoint = torch.load(model_path, map_location=device) # بارگذاری روی دیوایس صحیح
    
    # استخراج state_dict مدل. ممکن است مستقیما در checkpoint یا تحت کلید 'model' باشد.
    model_state_dict = checkpoint.get('model', checkpoint)

    # مدیریت بارگذاری state_dict اگر مدل با DataParallel ذخیره شده باشد
    if isinstance(model_eval, torch.nn.DataParallel):
         model_eval.load_state_dict(model_state_dict)
    else:
        # اگر مدل فعلی DataParallel نیست، اما state_dict ذخیره شده ممکن است باشد
        new_state_dict = {}
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k # حذف `module.` از ابتدای کلیدها
            new_state_dict[name] = v
        model_eval.load_state_dict(new_state_dict)

    model_eval = model_eval.to(device) # انتقال مدل به دیوایس (CPU/GPU)
    model_eval.eval() # تنظیم مدل به حالت ارزیابی (غیرفعال کردن dropout و batchnorm updates)

    hit_test, mrr_test = [], [] # لیست برای ذخیره نتایج Recall و MRR
    slices_test = test_data_loader.generate_batch(opt.batchSize) # تولید بچ‌های تست

    if not slices_test:
        print("هشدار: هیچ بچی از داده‌های تست نهایی تولید نشد. ارزیابی نهایی انجام نمی‌شود.")
        return 0.0, 0.0

    with torch.no_grad(): # غیرفعال کردن محاسبه گرادیان در حین ارزیابی
        for i_test_slice_indices in slices_test:
            if len(i_test_slice_indices) == 0:
                continue
            
            # دریافت داده‌های بچ فعلی از دیتا لودر تست
            # ssl_item_drop_prob=0.0 به این معنی است که هیچ آیتمی برای ارزیابی حذف نمی‌شود
            data_v1_test_tuple, _, targets_test_np, mask_test_np, time_diffs_test_np, _ = test_data_loader.get_slice(
                i_test_slice_indices, ssl_item_drop_prob=0.0
            )

            alias_inputs_test, A_test, items_test_unique, _, position_ids_test = data_v1_test_tuple

            if items_test_unique.size == 0:
                print("یک بچ تست به دلیل خالی بودن آرایه آیتم‌های یکتا نادیده گرفته شد.")
                continue

            # تبدیل داده‌های NumPy به تنسورهای PyTorch و انتقال به دیوایس
            items_test_unique = torch.from_numpy(items_test_unique).long().to(device)
            A_test = torch.from_numpy(A_test).float().to(device)
            alias_inputs_test = torch.from_numpy(alias_inputs_test).long().to(device)
            position_ids_test = torch.from_numpy(position_ids_test).long().to(device)
            time_diffs_test = torch.from_numpy(time_diffs_test_np).float().to(device)
            mask_test_cuda = torch.from_numpy(mask_test_np).long().to(device)
            targets_test_cuda = torch.from_numpy(targets_test_np).long().to(device)
            
            # ایجاد ماسک برای لایه MultiheadAttention (True برای پوزیشن‌های پد شده)
            seq_attention_mask_test = (mask_test_cuda == 0)

            # اجرای مدل روی داده‌های تست
            final_seq_hidden_test, star_context_test = model_eval(items_test_unique, A_test, alias_inputs_test, position_ids_test, time_diffs_test, seq_attention_mask_test)
            
            if final_seq_hidden_test.size(0) == 0:
                print("یک بچ تست به دلیل خالی بودن final_seq_hidden_test نادیده گرفته شد.")
                continue
            
            # محاسبه امتیازات برای آیتم‌های کاندید
            # بررسی اینکه آیا model_eval یک ماژول DataParallel است یا خیر
            if isinstance(model_eval, torch.nn.DataParallel):
                scores_test = model_eval.module.compute_scores(final_seq_hidden_test, star_context_test, mask_test_cuda)
            else:
                scores_test = model_eval.compute_scores(final_seq_hidden_test, star_context_test, mask_test_cuda)

            # دریافت 20 آیتم برتر پیش‌بینی شده
            sub_scores_top20_indices = scores_test.topk(20)[1]
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            # تنظیم اهداف (targets) برای مقایسه (0-ایندکس شده)
            targets_test_for_metric_np = targets_test_cuda.cpu().detach().numpy() - 1

            # محاسبه Recall@20 و MRR@20
            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_test_for_metric_np):
                hit_test.append(np.isin(target_item, score_row))
                if target_item in score_row:
                    rank = np.where(score_row == target_item)[0][0] + 1 # رتبه 1-مبنا
                    mrr_test.append(1.0 / rank)
                else:
                    mrr_test.append(0.0)

    # محاسبه میانگین Recall و MRR
    final_hit_metric = np.mean(hit_test) * 100 if hit_test else 0.0
    final_mrr_metric = np.mean(mrr_test) * 100 if mrr_test else 0.0

    print(f"نتایج مجموعه تست نهایی - Recall@20: {final_hit_metric:.4f}, MRR@20: {final_mrr_metric:.4f}")
    return final_hit_metric, final_mrr_metric


def main(opt):
    # تنظیم مسیرها
    data_dir = f'./{opt.dataset}/' # مسیر دیتاست (فرض می‌شود در روت و داخل پوشه با نام دیتاست است)
    model_save_dir = 'saved_star_models/' # مسیر ذخیره مدل‌ها
    log_dir = 'logs_star/' # مسیر لاگ‌های TensorBoard
    plot_dir = 'plots_star/' # مسیر ذخیره نمودارها

    # ایجاد دایرکتوری‌ها در صورت عدم وجود
    for directory in [model_save_dir, log_dir, plot_dir]: #
        if not os.path.exists(directory):
            os.makedirs(directory) #
            print(f"دایرکتوری {directory} ایجاد شد.")

    writer = SummaryWriter(log_dir=log_dir) # ایجاد شیء TensorBoard writer

    # تنظیم دیوایس (CPU یا GPU)
    device = torch.device("cuda" if torch.cuda.is_available() and opt.n_gpu > 0 else "cpu") #
    if torch.cuda.is_available() and opt.n_gpu > 0: #
        print(f"در حال استفاده از {torch.cuda.device_count()} GPU") #
    else:
        print("در حال استفاده از CPU") #

    # بارگذاری داده‌ها
    try:
        print(f"در حال بارگذاری داده از: {data_dir}")
        with open(os.path.join(data_dir, 'train.txt'), 'rb') as f: #
            train_data_raw_full = pickle.load(f) # کل داده‌های آموزشی خام
        with open(os.path.join(data_dir, 'test.txt'), 'rb') as f: #
            test_data_raw_orig = pickle.load(f) # داده‌های تست اصلی خام
        with open(os.path.join(data_dir, 'time_data.pkl'), 'rb') as f: #
            time_data_all = pickle.load(f) # داده‌های زمانی
    except Exception as e:
        print(f"خطا در بارگذاری داده: {e}") #
        sys.exit(1) #

    time_data_train_full = time_data_all['train_time_diffs'] # داده‌های زمانی برای کل آموزش
    time_data_test_orig = time_data_all['test_time_diffs'] # داده‌های زمانی برای تست اصلی

    # محاسبه n_node (تعداد آیتم‌های یکتا) بر اساس کل داده‌های آموزشی و تست اصلی
    n_node = get_n_node_from_data(train_data_raw_full, test_data_raw_orig)
    print(f"n_node (محاسبه شده پویا از کل داده آموزشی و تست اصلی): {n_node}")

    # لیست برای ذخیره تاریخچه لاس و متریک‌ها
    train_loss_history = []
    train_main_loss_history = []
    train_ssl_loss_history = []
    val_recall_history = []
    val_mrr_history = []
    
    best_recall_model_path = "" # مسیر بهترین مدل ذخیره شده بر اساس recall ولیدیشن

    # تقسیم داده آموزشی به آموزشی و ولیدیشن (اگر opt.validation=True باشد)
    if opt.validation:
        # ایجاد اندیس‌های shuffle شده برای تقسیم هماهنگ داده‌های اصلی و زمانی
        sidx = np.arange(len(train_data_raw_full[0]), dtype='int32')
        np.random.shuffle(sidx)
        n_train_samples = int(np.round(len(train_data_raw_full[0]) * (1. - opt.valid_portion)))

        # تقسیم داده‌های اصلی آموزشی
        train_data_for_loader = ([train_data_raw_full[0][i] for i in sidx[:n_train_samples]],
                                [train_data_raw_full[1][i] for i in sidx[:n_train_samples]])
        # داده‌های ولیدیشن (برای ارزیابی دوره‌ای)
        test_data_raw_for_eval = ([train_data_raw_full[0][i] for i in sidx[n_train_samples:]],
                                 [train_data_raw_full[1][i] for i in sidx[n_train_samples:]])

        # تقسیم داده‌های زمانی متناسب با داده‌های اصلی
        time_data_train = [time_data_train_full[i] for i in sidx[:n_train_samples]] #
        time_data_valid_for_eval = [time_data_train_full[i] for i in sidx[n_train_samples:]] #
        
    else:
        # اگر ولیدیشن نباشد، از کل داده آموزشی برای آموزش و از داده تست اصلی برای ارزیابی دوره‌ای استفاده می‌شود
        train_data_for_loader = train_data_raw_full
        test_data_raw_for_eval = test_data_raw_orig
        time_data_train = time_data_train_full
        time_data_valid_for_eval = time_data_test_orig

    # ایجاد دیتا لودرها
    train_data_loader = Dataset(train_data_for_loader, time_data_train, shuffle=True, opt=opt)
    eval_data_loader = Dataset(test_data_raw_for_eval, time_data_valid_for_eval, shuffle=False, opt=opt)
    
    # به‌روزرسانی max_len در opt بر اساس حداکثر طول واقعی در دیتا لودرها
    actual_dataset_max_len = max(train_data_loader.len_max, eval_data_loader.len_max)
    if opt.max_len == 0 or opt.max_len < actual_dataset_max_len:
        print(f"به‌روزرسانی opt.max_len از {opt.max_len} به {actual_dataset_max_len}")
        opt.max_len = actual_dataset_max_len
    
    # تنظیم ابعاد امبدینگ موقعیت و پروجکشن SSL اگر 0 باشند
    if opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
    if opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2

    # ایجاد نمونه مدل
    model_instance = Attention_SessionGraph(opt, n_node)

    # استفاده از DataParallel برای چند GPU (اگر موجود و تنظیم شده باشد)
    if opt.n_gpu > 1 and torch.cuda.is_available():
        print(f"استفاده از {opt.n_gpu} GPU با DataParallel")
        model = torch.nn.DataParallel(model_instance)
    else:
        model = model_instance
        
    model = model.to(device) # انتقال مدل به دیوایس

    print(f"مدل مقداردهی اولیه شد. n_node={n_node}, max_session_len={opt.max_len}, pos_emb_dim={opt.position_emb_dim}")
    print(f"هایپرپارامترها: {vars(opt)}")

    start_time = time.time() # زمان شروع آموزش
    best_result = [0.0, 0.0] # [بهترین recall ولیدیشن, بهترین mrr ولیدیشن]
    best_epoch = [0, 0] # اپوکی که بهترین نتایج در آن به دست آمده
    bad_counter = 0 # شمارنده برای early stopping

    # حلقه آموزش بر اساس تعداد اپوک‌ها
    for epoch_num in range(opt.epoch):
        print('-' * 50 + f'\nاپوک: {epoch_num}')
        opt.current_epoch_num = epoch_num # ذخیره شماره اپوک فعلی در opt برای استفاده در train_test
        
        # اجرای یک اپوک آموزش و ارزیابی روی داده ولیدیشن
        # تابع train_test باید میانگین لاس‌های آموزشی را نیز برگرداند
        eval_hit, eval_mrr, avg_train_total_loss, avg_train_main_loss, avg_train_ssl_loss = train_test(
            model, train_data_loader, eval_data_loader, opt, device
        )

        # بررسی اینکه آیا train_test مقادیر معتبری برگردانده است
        if avg_train_total_loss is None or np.isnan(avg_train_total_loss):
            print(f"هشدار: train_test لاس نامعتبر برای اپوک {epoch_num} برگرداند. لاگ کردن متریک‌ها و ذخیره مدل برای این اپوک انجام نمی‌شود.")
        else:
            print(f'اپوک {epoch_num} ارزیابی ولیدیشن - Recall@20: {eval_hit:.4f}, MRR@20: {eval_mrr:.4f}')
            print(f'اپوک {epoch_num} میانگین لاس آموزش - کل: {avg_train_total_loss:.4f}, اصلی: {avg_train_main_loss:.4f}, SSL: {avg_train_ssl_loss:.4f}')

            # لاگ کردن متریک‌ها در TensorBoard
            writer.add_scalar('epoch/eval_recall_at_20', eval_hit, epoch_num)
            writer.add_scalar('epoch/eval_mrr_at_20', eval_mrr, epoch_num)
            writer.add_scalar('epoch/train_total_loss', avg_train_total_loss, epoch_num)
            writer.add_scalar('epoch/train_main_loss', avg_train_main_loss, epoch_num)
            writer.add_scalar('epoch/train_ssl_loss', avg_train_ssl_loss, epoch_num)
            
            # ذخیره تاریخچه لاس و متریک‌ها
            train_loss_history.append(avg_train_total_loss)
            train_main_loss_history.append(avg_train_main_loss)
            train_ssl_loss_history.append(avg_train_ssl_loss)
            val_recall_history.append(eval_hit)
            val_mrr_history.append(eval_mrr)

            flag = 0 # پرچم برای بررسی اینکه آیا در این اپوک بهبودی حاصل شده است
            
            # بررسی و ذخیره بهترین مدل بر اساس Recall ولیدیشن
            if eval_hit >= best_result[0]:
                best_result[0] = eval_hit
                best_epoch[0] = epoch_num
                flag = 1
                
                # مسیر ثابت برای بهترین مدل recall
                current_best_recall_path = os.path.join(model_save_dir, f'best_recall_model.pt')
                # اگر فایل قبلی وجود داشت و مسیر جدید همان است، می‌توان آن را حذف کرد (اختیاری، چون بازنویسی می‌شود)
                # یا اینکه قبل از ذخیره، فایل موجود با این نام حذف شود.
                if os.path.exists(current_best_recall_path):
                    try:
                        os.remove(current_best_recall_path)
                    except OSError as e:
                        print(f"خطا در حذف مدل بهترین recall قبلی: {e}")
                
                state_to_save = {
                    'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'opt': opt,
                    'epoch': epoch_num,
                    'recall': eval_hit,
                    'mrr': eval_mrr
                }
                try:
                    torch.save(state_to_save, current_best_recall_path)
                    best_recall_model_path = current_best_recall_path # به‌روزرسانی مسیر بهترین مدل
                    print(f"بهترین مدل (بر اساس Recall ولیدیشن) در {best_recall_model_path} ذخیره شد.")
                except Exception as e:
                    print(f"خطا در ذخیره بهترین مدل recall: {e}")

            # بررسی و ذخیره بهترین مدل بر اساس MRR ولیدیشن (به صورت جداگانه)
            if eval_mrr >= best_result[1]:
                best_result[1] = eval_mrr
                best_epoch[1] = epoch_num
                flag = 1 # اگر recall همزمان بهترین شده باشد، flag از قبل 1 است
                
                current_best_mrr_path = os.path.join(model_save_dir, f'best_mrr_model.pt')
                if os.path.exists(current_best_mrr_path):
                     try:
                        os.remove(current_best_mrr_path)
                     except OSError as e:
                        print(f"خطا در حذف مدل بهترین MRR قبلی: {e}")
                
                state_to_save = {
                    'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'opt': opt,
                    'epoch': epoch_num,
                    'recall': eval_hit, # recall و mrr مربوط به همین اپوک ذخیره می‌شوند
                    'mrr': eval_mrr
                }
                try:
                    torch.save(state_to_save, current_best_mrr_path)
                    print(f"بهترین مدل (بر اساس MRR ولیدیشن) در {current_best_mrr_path} ذخیره شد.")
                except Exception as e:
                    print(f"خطا در ذخیره بهترین مدل MRR: {e}")

            print(f'بهترین نتیجه فعلی ولیدیشن: Recall@20: {best_result[0]:.4f} (اپوک {best_epoch[0]}), MRR@20: {best_result[1]:.4f} (اپوک {best_epoch[1]})')
            
            # مدیریت early stopping
            bad_counter = 0 if flag else bad_counter + 1
            if bad_counter >= opt.patience:
                print(f"توقف زودهنگام پس از {opt.patience} اپوک بدون بهبود در مجموعه ولیدیشن.")
                break # خروج از حلقه آموزش
            
    writer.close() # بستن TensorBoard writer
    print('-' * 50 + f"\nکل زمان آموزش/ولیدیشن: {(time.time() - start_time)/3600:.2f} ساعت")

    # --- ارزیابی نهایی روی مجموعه داده تست اصلی با بهترین مدل (بر اساس recall ولیدیشن) ---
    if best_recall_model_path and os.path.exists(best_recall_model_path):
        # ایجاد یک کپی از opt برای اطمینان از عدم تغییر پارامترهای اصلی opt
        # n_node و سایر پارامترهای مهم باید با مقادیری که مدل با آن‌ها آموزش دیده، یکسان باشند.
        opt_for_test_final = argparse.Namespace(**vars(opt))
        
        # اطمینان از اینکه max_len برای دیتا لودر تست نهایی صحیح است
        # این مقدار باید با max_len استفاده شده در زمان آموزش مدل بهترین recall هماهنگ باشد.
        # اگر checkpoint شامل opt باشد، می‌توان از آن استفاده کرد.
        # در اینجا فرض می‌کنیم opt.max_len در طول آموزش به‌روز شده و صحیح است.
        
        # ایجاد دیتا لودر برای تست نهایی با استفاده از داده‌های تست اصلی
        test_final_loader = Dataset(test_data_raw_orig, time_data_test_orig, shuffle=False, opt=opt_for_test_final)
        
        # اگر max_len در test_final_loader با opt.max_len اصلی متفاوت شد (نباید بشود اگر opt درست منتقل شود)
        # می‌توانیم آن را برای evaluate_on_test_set به‌روز کنیم.
        if opt_for_test_final.max_len < test_final_loader.len_max:
             print(f"هشدار: opt.max_len برای لودر تست نهایی ({opt_for_test_final.max_len}) کمتر از حداکثر طول واقعی در داده تست ({test_final_loader.len_max}) است. در حال تنظیم مجدد.")
             opt_for_test_final.max_len = test_final_loader.len_max

        evaluate_on_test_set(best_recall_model_path, test_final_loader, opt_for_test_final, device, n_node)
    else:
        print("هیچ مدل بهتری (بر اساس recall) ذخیره یا پیدا نشد. ارزیابی نهایی روی مجموعه تست انجام نمی‌شود.")


    # --- رسم نمودارها ---
    # اطمینان از اینکه تاریخچه‌ها خالی نیستند قبل از رسم نمودار
    if not train_loss_history or not val_recall_history:
        print("داده‌ای برای رسم نمودار وجود ندارد. از رسم نمودار صرف‌نظر می‌شود.")
        return # اگر داده‌ای نباشد، از ادامه تابع خارج می‌شویم

    epochs_range = range(len(train_loss_history)) # محدوده اپوک‌ها برای محور x

    plt.figure(figsize=(18, 10)) # اندازه کلی شکل

    # نمودار لاس کل آموزش
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_loss_history, label='Train Total Loss')
    plt.title('Training Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # نمودار لاس اصلی آموزش
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, train_main_loss_history, label='Train Main Loss')
    plt.title('Training Main Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # نمودار لاس SSL آموزش
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, train_ssl_loss_history, label='Train SSL Loss')
    plt.title('Training SSL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # نمودار Recall@20 ولیدیشن
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, val_recall_history, label='Validation Recall@20')
    plt.title('Recall@20')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@20')
    plt.legend()

    # نمودار MRR@20 ولیدیشن
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, val_mrr_history, label='Validation MRR@20')
    plt.title('MRR@20')
    plt.xlabel('Epoch')
    plt.ylabel('MRR@20')
    plt.legend()
    
    # نمودار بررسی Overfitting (Recall ولیدیشن در مقابل لاس آموزش)
    plt.subplot(2, 3, 6)
    ax1 = plt.gca() # گرفتن محور فعلی
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Recall@20', color=color)
    ax1.plot(epochs_range, val_recall_history, color=color, label='Val Recall@20')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx() # ایجاد یک محور y دوم که محور x یکسانی دارد
    color = 'tab:blue'
    ax2.set_ylabel('Train Total Loss', color=color)
    ax2.plot(epochs_range, train_loss_history, color=color, linestyle='--', label='Train Total Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    plt.title('Overfitting Check: Val Recall vs. Train Loss')


    plt.tight_layout() # تنظیم چیدمان نمودارها برای جلوگیری از همپوشانی
    # اطمینان از اینکه epoch_num مقدار آخرین اپوک اجرا شده را دارد
    last_epoch_ran = epoch_num if 'epoch_num' in locals() else opt.epoch -1
    plot_filename = os.path.join(plot_dir, f'{opt.dataset}_training_plots_epoch_{last_epoch_ran}.png')
    plt.savefig(plot_filename) # ذخیره نمودارها در فایل
    print(f"نمودارهای آموزش در {plot_filename} ذخیره شدند.")
    # plt.show() # نمایش نمودارها (در محیط بدون GUI کامنت شود)


if __name__ == '__main__':
    # پارسر آرگومان‌های خط فرمان
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # تعریف آرگومان‌های قابل تنظیم از طریق خط فرمان
    parser.add_argument('--dataset', default='yoochoose1_64', choices=['diginetica', 'yoochoose1_64'], help='نام دیتاست')
    parser.add_argument('--validation', type=str2bool, default=True, help='استفاده از تقسیم ولیدیشن')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='نسبت داده ولیدیشن')

    parser.add_argument('--hiddenSize', type=int, default=512, help='ابعاد لایه پنهان')
    parser.add_argument('--step', type=int, default=4, help='تعداد گام‌های انتشار GNN')
    parser.add_argument('--nonhybrid', type=str2bool, default=False, help='استفاده از امتیازدهی غیرهیبریدی')
    parser.add_argument('--max_len', type=int, default=0, help='حداکثر طول سکانس (0=خودکار)')
    parser.add_argument('--position_emb_dim', type=int, default=0, help='ابعاد امبدینگ موقعیت (0=hiddenSize)')

    parser.add_argument('--n_gpu', type=int, default=1, help='تعداد GPU (0=CPU)')
    parser.add_argument('--batchSize', type=int, default=512, help='اندازه بچ')
    parser.add_argument('--epoch', type=int, default=100, help='تعداد اپوک‌ها')
    parser.add_argument('--lr', type=float, default=0.001, help='نرخ یادگیری')
    parser.add_argument('--l2', type=float, default=1e-4, help='ضریب L2 penalty')
    parser.add_argument('--patience', type=int, default=20, help='تعداد اپوک برای early stopping')

    parser.add_argument('--ssl_weight', type=float, default=0.5, help='وزن لاس SSL')
    parser.add_argument('--ssl_temperature', type=float, default=0.1, help='دمای SSL')
    parser.add_argument('--ssl_item_drop_prob', type=float, default=0.4, help='احتمال حذف آیتم در SSL')
    parser.add_argument('--ssl_projection_dim', type=int, default=0, help='ابعاد لایه پروجکشن SSL (0=hiddenSize/2)')

    cmd_args = parser.parse_args() # خواندن آرگومان‌های ورودی
    
    opt = argparse.Namespace() # ایجاد یک Namespace برای نگهداری تنظیمات نهایی
    
    # بارگذاری تنظیمات پیش‌فرض بر اساس نام دیتاست
    if cmd_args.dataset == 'diginetica':
        base_config = Diginetica_arg()
    elif cmd_args.dataset == 'yoochoose1_64':
        base_config = Yoochoose_arg()
    else:
        print(f"خطا: دیتاست ناشناخته '{cmd_args.dataset}'")
        sys.exit(1)
    
    # کپی کردن تنظیمات پیش‌فرض به opt
    for key, value in vars(base_config).items():
        setattr(opt, key, value)
    
    # بازنویسی تنظیمات پیش‌فرض با مقادیر وارد شده از خط فرمان (اگر وارد شده باشند)
    for key, value in vars(cmd_args).items():
        if hasattr(opt, key) and value is not None: # فقط اگر مقدار جدیدی از cmd وارد شده باشد
            setattr(opt, key, value)
    
    # تنظیم مقادیر پیش‌فرض نهایی برای ssl_projection_dim و position_emb_dim اگر هنوز 0 هستند
    if opt.ssl_projection_dim == 0:
        opt.ssl_projection_dim = opt.hiddenSize // 2
    if opt.position_emb_dim == 0:
        opt.position_emb_dim = opt.hiddenSize
        
    print("پیکربندی نهایی:")
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    
    main(opt) # اجرای تابع اصلی برنامه
