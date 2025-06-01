import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from tqdm import tqdm
from collections import defaultdict # برای item_dict و cat_dict بهتر است

# تنظیمات شبیه‌سازی شده از opt
class OptConfig:
    dataset = "yoochoose" # یا 'diginetica'
    # سایر تنظیمات مورد نیاز مانند مسیر دیتاست و خروجی
    yoochoose_dat_file = 'yoochoose/yoochoose-clicks.dat' # مسیر فایل .dat شما
    output_dir_yoochoose = 'yoochoose1_64' # مطابق با مدل اصلی
    output_dir_diginetica = 'diginetica'

opt = OptConfig()

# تعیین مسیر دیتاست بر اساس opt.dataset
if opt.dataset == 'yoochoose':
    dataset_file_path = opt.yoochoose_dat_file
    output_data_dir = opt.output_dir_yoochoose
elif opt.dataset == 'diginetica':
    # dataset_file_path = 'مسیر فایل دیتاست دیجی‌نتیکا شما'
    # output_data_dir = opt.output_dir_diginetica
    print(f"Dataset {opt.dataset} configuration not fully implemented in this example.")
    exit()
else:
    print(f"Unknown dataset: {opt.dataset}")
    exit()

print(f"-- Starting @ {datetime.datetime.now()} for dataset: {opt.dataset}")

# ایجاد دایرکتوری خروجی در صورت عدم وجود
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)
    print(f"Created directory: {output_data_dir}")

# برای جلوگیری از خواندن مجدد فایل بزرگ در هر بار اجرا (اختیاری)
temp_pickle_file = os.path.join(output_data_dir, "temp_processed_clicks.pkl")

try:
    sess_clicks_orig_format, sess_last_timestamp = pickle.load(open(temp_pickle_file, "rb"))
    print("Loaded saved intermediate pickle for clicks and timestamps.")
except FileNotFoundError:
    print(f"Intermediate pickle not found at {temp_pickle_file}. Processing from raw data...")
    sess_clicks_orig_format = defaultdict(list) # session_id: [(item_id, category_id, timestamp_float), ...]
    sess_last_timestamp = {} # session_id: last_event_timestamp_float (برای تقسیم train/test)

    with open(dataset_file_path, "r") as f:
        if opt.dataset == 'yoochoose':
            # فرمت Yoochoose: session_id,timestamp,item_id,category
            # ستون‌ها بدون هدر هستند.
            reader = csv.reader(f, delimiter=',')
            header = ['session_id', 'timestamp', 'item_id', 'category'] # تعریف هدر دستی
        else: # فرض کنید سایر دیتاست‌ها هدر دارند
            reader = csv.DictReader(f, delimiter=';') # مثال برای Diginetica

        for row_idx, row_data in enumerate(tqdm(reader, desc="Reading raw data")):
            try:
                if opt.dataset == 'yoochoose':
                    # تبدیل row_data (لیست) به دیکشنری با استفاده از هدر
                    data = {header[i]: row_data[i] for i in range(len(row_data))}
                    if len(row_data) != 4: # بررسی تعداد ستون‌ها برای Yoochoose
                        # print(f"Skipping malformed row (Yoochoose): {row_data}")
                        continue
                else:
                    data = row_data # برای DictReader

                session_id = data['session_id']
                item_id_str = data['item_id']
                
                if opt.dataset == 'yoochoose':
                    category_str = data['category'] # می‌تواند 'S' یا یک عدد یا ۰ باشد
                    timestamp_str = data['timestamp'] # فرمت: 2014-04-01T03:00:00.123Z
                    # تبدیل timestamp به float
                    dt_object = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                    timestamp_float = dt_object.timestamp()
                else: # Diginetica example
                    category_str = "0" # Diginetica ممکن است category نداشته باشد یا متفاوت باشد
                    # timestamp_float = float(data['timeframe']) # یا data['eventdate']
                    print("Timestamp handling for non-Yoochoose dataset needs specific implementation.")
                    continue # برای این مثال، فقط Yoochoose را کامل پیاده‌سازی می‌کنیم

                sess_clicks_orig_format[session_id].append((item_id_str, category_str, timestamp_float))
                
                # به‌روزرسانی آخرین timestamp برای هر جلسه
                if session_id not in sess_last_timestamp or timestamp_float > sess_last_timestamp[session_id]:
                    sess_last_timestamp[session_id] = timestamp_float

            except Exception as e:
                # print(f"Error processing row {row_idx+1}: {row_data} -> {e}")
                continue
    
    # مرتب‌سازی کلیک‌ها در هر جلسه بر اساس زمان (اگر قبلاً انجام نشده)
    for session_id in tqdm(sess_clicks_orig_format, desc="Sorting session clicks by time"):
        sess_clicks_orig_format[session_id].sort(key=operator.itemgetter(2)) # Sort by timestamp_float

    pickle.dump((sess_clicks_orig_format, sess_last_timestamp), open(temp_pickle_file, "wb"))
    print(f"Saved intermediate pickle to {temp_pickle_file}")

print(f"-- Reading data complete @ {datetime.datetime.now()}")
print(f"Total unique sessions read: {len(sess_clicks_orig_format)}")

# 1. فیلتر کردن سشن‌های با طول ۱
print("Filtering sessions with length 1...")
sessions_to_delete = [s for s in sess_clicks_orig_format if len(sess_clicks_orig_format[s]) <= 1]
for s in sessions_to_delete:
    del sess_clicks_orig_format[s]
    if s in sess_last_timestamp:
        del sess_last_timestamp[s]
print(f"Sessions after filtering length 1: {len(sess_clicks_orig_format)}")

# 2. شمارش تعداد تکرار هر آیتم و فیلتر آیتم‌های نادر (کمتر از ۵ بار تکرار)
print("Counting item occurrences and filtering rare items...")
item_counts = defaultdict(int)
for session_id in sess_clicks_orig_format:
    for item_tuple in sess_clicks_orig_format[session_id]:
        item_counts[item_tuple[0]] += 1

# فیلتر کردن آیتم‌های نادر از سشن‌ها
# و سپس فیلتر کردن سشن‌هایی که پس از حذف آیتم‌های نادر، طولشان کمتر از ۲ شده
sessions_to_delete_after_item_filter = []
for session_id in list(sess_clicks_orig_format.keys()): # Iterate over a copy of keys for safe deletion
    original_session_tuples = sess_clicks_orig_format[session_id]
    # نگه داشتن آیتم‌هایی که تعدادشان >= 5 است
    filtered_session_tuples = [item_tuple for item_tuple in original_session_tuples if item_counts[item_tuple[0]] >= 5]
    
    if len(filtered_session_tuples) < 2: # اگر طول سشن پس از فیلتر کمتر از ۲ شد
        sessions_to_delete_after_item_filter.append(session_id)
    else:
        sess_clicks_orig_format[session_id] = filtered_session_tuples
        # به‌روزرسانی sess_last_timestamp اگر سشن تغییر کرده (مهم نیست اگر فقط آیتم حذف شده)

for s in sessions_to_delete_after_item_filter:
    if s in sess_clicks_orig_format:
        del sess_clicks_orig_format[s]
    if s in sess_last_timestamp:
        del sess_last_timestamp[s]
print(f"Sessions after filtering rare items and subsequent short sessions: {len(sess_clicks_orig_format)}")


# 3. تقسیم داده‌ها به آموزشی و آزمایشی بر اساس تاریخ
# در کد شما برای Yoochoose، یک روز آخر برای تست بود. در agc.py اصلی ۷ روز بود. من از ۱ روز استفاده می‌کنم.
dates_for_split = list(sess_last_timestamp.items()) # [(session_id, last_timestamp_float), ...]
if not dates_for_split:
    print("No sessions left after filtering. Exiting.")
    exit()

max_timestamp = max(ts for _, ts in dates_for_split)

# برای Yoochoose، ۱ روز آخر را به عنوان تست در نظر می‌گیریم
split_timestamp = max_timestamp - (86400 * 1) # 1 day in seconds

train_session_ids_with_ts = sorted(filter(lambda x: x[1] < split_timestamp, dates_for_split), key=operator.itemgetter(1))
test_session_ids_with_ts = sorted(filter(lambda x: x[1] >= split_timestamp, dates_for_split), key=operator.itemgetter(1)) # >= برای پوشش دقیق

train_session_ids = [s_id for s_id, _ in train_session_ids_with_ts]
test_session_ids = [s_id for s_id, _ in test_session_ids_with_ts]

print(f"Train sessions: {len(train_session_ids)}")
print(f"Test sessions: {len(test_session_ids)}")
print(f"-- Splitting train and test sets @ {datetime.datetime.now()}")

# 4. ساخت دیکشنری آیتم‌ها (item_dict) فقط بر اساس داده‌های آموزشی
# و تبدیل سکوئنس‌ها به فرمت مورد نیاز مدل قبلی
item_map = {} # item_id_str -> new_int_id (starting from 1)
current_item_id = 1

def process_sessions_for_output(session_ids_list, is_training_set):
    global current_item_id # برای به‌روزرسانی item_map در حین پردازش ترین
    
    output_sequences = [] # لیست سکوئنس‌های آیتم (اعداد صحیح)
    output_targets = []   # لیست آیتم‌های هدف (اعداد صحیح)
    output_time_diffs = [] # لیست لیست‌های اختلاف زمانی

    for session_id in tqdm(session_ids_list, desc=f"Processing {'train' if is_training_set else 'test'} sessions"):
        session_tuples = sess_clicks_orig_format[session_id] # [(item_id_str, cat_str, ts_float), ...]
        
        current_sequence_item_ids_int = []
        current_sequence_timestamps_float = []

        # حذف آیتم‌های متوالی تکراری و نگاشت آیتم‌ها
        last_item_id_str = None
        for item_id_str, _, timestamp_float in session_tuples:
            if item_id_str == last_item_id_str: # حذف تکراری‌های متوالی
                continue
            
            if is_training_set:
                if item_id_str not in item_map:
                    item_map[item_id_str] = current_item_id
                    current_item_id += 1
                current_sequence_item_ids_int.append(item_map[item_id_str])
                current_sequence_timestamps_float.append(timestamp_float)
            else: # برای تست، فقط آیتم‌هایی که در ترین بوده‌اند
                if item_id_str in item_map:
                    current_sequence_item_ids_int.append(item_map[item_id_str])
                    current_sequence_timestamps_float.append(timestamp_float)
            last_item_id_str = item_id_str
        
        if len(current_sequence_item_ids_int) < 2: # اگر طول سکوئنس پس از پردازش‌ها کمتر از ۲ شد
            continue

        # ساخت ورودی و هدف برای مدل قبلی
        # input_seq = [s1, s2, ..., s_{n-1}], target = sn
        input_seq_for_model = current_sequence_item_ids_int[:-1]
        target_for_model = current_sequence_item_ids_int[-1]
        
        # محاسبه اختلاف زمانی برای input_seq_for_model
        time_diffs_for_seq = [0.0] # اولین آیتم اختلاف زمانی صفر دارد
        if len(input_seq_for_model) > 1:
            timestamps_for_input_seq = current_sequence_timestamps_float[:-1] # timestampهای متناظر با input_seq_for_model
            for i in range(len(timestamps_for_input_seq) - 1):
                diff = timestamps_for_input_seq[i+1] - timestamps_for_input_seq[i]
                # اختلاف زمانی نباید منفی باشد، اگر بود مشکلی وجود دارد (مثلا مرتب‌سازی اولیه اشتباه بوده)
                # یا خطای دقت float. برای سادگی، مقادیر کوچک منفی را صفر می‌کنیم.
                time_diffs_for_seq.append(max(0.0, diff)) 

        output_sequences.append(input_seq_for_model)
        output_targets.append(target_for_model)
        output_time_diffs.append(time_diffs_for_seq)
        
    return output_sequences, output_targets, output_time_diffs

print("Processing training data...")
train_sequences, train_targets, train_time_diffs = process_sessions_for_output(train_session_ids, True)
print(f"Number of items in item_map (from training data): {len(item_map)}")
print(f"Max item_id created: {current_item_id -1}")


print("Processing testing data...")
test_sequences, test_targets, test_time_diffs = process_sessions_for_output(test_session_ids, False)

# 5. ذخیره‌سازی داده‌ها با فرمت مورد نیاز
# train.txt: (list_of_train_sequences, list_of_train_targets)
train_data_to_save = (train_sequences, train_targets)
train_file_path = os.path.join(output_data_dir, 'train.txt')
with open(train_file_path, 'wb') as f_train:
    pickle.dump(train_data_to_save, f_train)
print(f"Saved training data to {train_file_path}")

# test.txt: (list_of_test_sequences, list_of_test_targets)
test_data_to_save = (test_sequences, test_targets)
test_file_path = os.path.join(output_data_dir, 'test.txt')
with open(test_file_path, 'wb') as f_test:
    pickle.dump(test_data_to_save, f_test)
print(f"Saved testing data to {test_file_path}")

# time_data.pkl: {'train_time_diffs': train_time_diffs, 'test_time_diffs': test_time_diffs}
time_data_to_save = {
    'train_time_diffs': train_time_diffs,
    'test_time_diffs': test_time_diffs
}
time_data_file_path = os.path.join(output_data_dir, 'time_data.pkl')
with open(time_data_file_path, 'wb') as f_time:
    pickle.dump(time_data_to_save, f_time)
print(f"Saved time data to {time_data_file_path}")


print(f"-- Data processing and saving finished @ {datetime.datetime.now()}")
print(f"Final number of training sequences: {len(train_sequences)}")
print(f"Final number of testing sequences: {len(test_sequences)}")
print(f"Example train sequence: {train_sequences[0] if train_sequences else 'N/A'}")
print(f"Example train target: {train_targets[0] if train_targets else 'N/A'}")
print(f"Example train time_diffs: {train_time_diffs[0] if train_time_diffs else 'N/A'}")
print("Done.")