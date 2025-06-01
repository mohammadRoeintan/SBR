#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified in June 2025 based on user-provided sample logic for Yoochoose dataset
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: yoochoose')
# افزودن آرگومان برای مسیر فایل موقت، مشابه کد نمونه
parser.add_argument('--temp_pickle_path', default='./yoochoose_processed/temp_processing.pkl', help='Path for temporary processing pickle file')
parser.add_argument('--output_dir', default='yoochoose1_64_new', help='Directory for final output files')

opt = parser.parse_args()
print(opt)

# تنظیمات خاص دیتاست Yoochoose
if opt.dataset == 'yoochoose':
    dataset_file_path = '/kaggle/input/yoochoose/yoochoose-clicks.dat' # یا مسیر فایل شما
    # تعریف نام ستون‌ها بر اساس مستندات (اگر فایل هدر ندارد)
    # کد نمونه شما از DictReader استفاده می‌کند که هدر را می‌خواند.
    # اگر فایل yoochoose-clicks.dat شما هدر ندارد، باید ستون‌ها را مشخص کنید.
    # فرض می‌کنیم فایل هدر دارد یا فرمت آن مشابه چیزی است که DictReader انتظار دارد.
    # ستون‌های مورد انتظار در کد نمونه: 'session_id', 'timestamp', 'item_id', 'category_id' (در کد شما 'category')

print("-- Starting @ %s" % datetime.datetime.now())

# ایجاد دایرکتوری برای فایل موقت و خروجی اگر وجود ندارد
os.makedirs(os.path.dirname(opt.temp_pickle_path), exist_ok=True)
os.makedirs(opt.output_dir, exist_ok=True)

try:
    sess_clicks_ts_cat, sess_last_event_date = pickle.load(open(opt.temp_pickle_path, "rb"))
    print(f"Loaded saved intermediate pickle from {opt.temp_pickle_path}")
except FileNotFoundError:
    print(f"Temporary pickle not found at {opt.temp_pickle_path}. Processing from raw data...")
    sess_clicks_ts_cat = {}  # {session_id: [(item_id, category, timestamp_float), ...]}
    sess_last_event_date = {}    # {session_id: timestamp_float_of_last_event}
    
    # خواندن فایل دیتاست
    with open(dataset_file_path, "r") as f:
        # کد نمونه از DictReader استفاده می‌کند. اگر فایل شما هدر ندارد، باید reader را تغییر دهید.
        # فرض می‌کنیم فایل yoochoose-clicks.dat شما ستون‌های session_id,timestamp,item_id,category را دارد.
        reader = csv.reader(f, delimiter=',')
        header = next(reader) # خواندن و ذخیره هدر
        # پیدا کردن اندیس ستون‌ها
        try:
            idx_session = header.index('session_id')
            idx_ts = header.index('timestamp')
            idx_item = header.index('item_id')
            idx_cat = header.index('category') # یا 'category_id' اگر نام ستون این است
        except ValueError as e:
            print(f"Error: One or more expected columns not found in header: {header}")
            print(f"Expected 'session_id', 'timestamp', 'item_id', 'category'. Error: {e}")
            exit()

        for row in tqdm(reader):
            session_id = row[idx_session]
            timestamp_str = row[idx_ts]
            item_id = row[idx_item]
            category = row[idx_cat]

            try:
                # تبدیل timestamp به float (ثانیه از epoch)
                # فرمت Yoochoose: 2014-04-01T10:00:29.873Z
                dt_object = datetime.datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                timestamp_float = dt_object.timestamp()
            except ValueError:
                try:
                    # امتحان کردن فرمت بدون میلی‌ثانیه اگر اولی ناموفق بود
                    dt_object = datetime.datetime.strptime(timestamp_str[:19], '%Y-%m-%dT%H:%M:%S')
                    timestamp_float = dt_object.timestamp()
                except ValueError:
                    print(f"Warning: Could not parse timestamp '{timestamp_str}' for session {session_id}. Skipping event.")
                    continue
            
            if session_id not in sess_clicks_ts_cat:
                sess_clicks_ts_cat[session_id] = []
            sess_clicks_ts_cat[session_id].append((item_id, category, timestamp_float))
            
            # به‌روزرسانی آخرین تاریخ رویداد برای جلسه
            if session_id not in sess_last_event_date or timestamp_float > sess_last_event_date[session_id]:
                sess_last_event_date[session_id] = timestamp_float

    # مرتب‌سازی کلیک‌ها در هر جلسه بر اساس زمان (مطابق با منطق کد نمونه که در نهایت مرتب‌سازی انجام می‌دهد)
    for session_id in tqdm(sess_clicks_ts_cat):
        sess_clicks_ts_cat[session_id].sort(key=operator.itemgetter(2)) # مرتب‌سازی بر اساس timestamp_float

    pickle.dump((sess_clicks_ts_cat, sess_last_event_date), open(opt.temp_pickle_path, "wb"))
    print(f"Saved intermediate processing to {opt.temp_pickle_path}")

print("-- Reading data @ %s" % datetime.datetime.now())

# فیلتر کردن جلسات با طول ۱
print(f"Sessions before length 1 filter: {len(sess_clicks_ts_cat)}")
for s_id in list(sess_clicks_ts_cat.keys()):
    if len(sess_clicks_ts_cat[s_id]) == 1:
        del sess_clicks_ts_cat[s_id]
        del sess_last_event_date[s_id]
print(f"Sessions after length 1 filter: {len(sess_clicks_ts_cat)}")

# شمارش تعداد تکرار هر آیتم
item_counts = defaultdict(int)
for s_id in sess_clicks_ts_cat:
    for item_tuple in sess_clicks_ts_cat[s_id]:
        item_counts[item_tuple[0]] += 1

# فیلتر آیتم‌های نادر (کمتر از MIN_ITEM_COUNT بار ظاهر شده) و جلسات کوتاه‌تر از MIN_SESSION_LENGTH
MIN_ITEM_COUNT = 5
MIN_SESSION_LENGTH = 2 # کد نمونه ۲ را در نظر می‌گیرد

print(f"Sessions before rare item filter: {len(sess_clicks_ts_cat)}")
for s_id in list(sess_clicks_ts_cat.keys()):
    # ابتدا آیتم‌های نادر را فیلتر می‌کنیم
    filtered_session_events = [event for event in sess_clicks_ts_cat[s_id] if item_counts[event[0]] >= MIN_ITEM_COUNT]
    
    if len(filtered_session_events) < MIN_SESSION_LENGTH:
        del sess_clicks_ts_cat[s_id]
        del sess_last_event_date[s_id]
    else:
        sess_clicks_ts_cat[s_id] = filtered_session_events
print(f"Sessions after rare item filter and length check: {len(sess_clicks_ts_cat)}")


# تقسیم داده به آموزش/آزمون بر اساس تاریخ
# در کد نمونه، برای Yoochoose یک روز آخر برای آزمون است
dates_for_split = list(sess_last_event_date.items()) # [(session_id, last_timestamp_float), ...]
if not dates_for_split:
    print("Error: No session data remaining after filtering. Exiting.")
    exit()
    
max_date_overall = dates_for_split[0][1]
for _, date_val in dates_for_split:
    if max_date_overall < date_val:
        max_date_overall = date_val

# برای Yoochoose، یک روز آخر (86400 ثانیه)
split_date_threshold = max_date_overall - (86400 * 1) 
# split_date_threshold = max_date_overall - (86400 * 7) # برای 7 روز اگر دیتاست متفاوت باشد

print(f'Splitting date threshold: {split_date_threshold} (Timestamp corresponding to {datetime.datetime.fromtimestamp(split_date_threshold)})')

train_session_ids_dates = [] # لیستی از (session_id, last_timestamp_float)
test_session_ids_dates = []

for s_id, last_ts in dates_for_split:
    if last_ts < split_date_threshold:
        train_session_ids_dates.append((s_id, last_ts))
    else: # کد نمونه فقط > را در نظر می‌گیرد، اما >= منطقی‌تر است اگر بخواهیم دقیقاً از یک نقطه جدا کنیم
        test_session_ids_dates.append((s_id, last_ts))

# مرتب‌سازی جلسات بر اساس تاریخ (آخرین رویداد)
train_session_ids_dates.sort(key=operator.itemgetter(1))
test_session_ids_dates.sort(key=operator.itemgetter(1))

print(f"Total train sessions: {len(train_session_ids_dates)}")
print(f"Total test sessions: {len(test_session_ids_dates)}")
print("-- Splitting train and test set @ %s" % datetime.datetime.now())

# تابع حذف آیتم‌های متوالی تکراری
def delete_consecutive_duplicates(session_events):
    if not session_events:
        return []
    
    # session_events: [(item_id, category, timestamp_float), ...]
    # در اینجا فقط بر اساس item_id تکراری‌ها را حذف می‌کنیم، مشابه کد نمونه
    
    # ابتدا باید به فرمت مورد انتظار delete_dups در کد نمونه برسانیم: (لیست آیتم‌ها، لیست کتگوری‌ها، لیست تایم‌استمپ‌ها)
    items = [event[0] for event in session_events]
    cats = [event[1] for event in session_events] # نگه می‌داریم هرچند در مدل نهایی استفاده نشود
    timestamps = [event[2] for event in session_events]

    if not items:
        return []

    last_item = items[0]
    to_keep_indices = [0]
    for i in range(1, len(items)):
        if items[i] != last_item:
            last_item = items[i]
            to_keep_indices.append(i)
            
    # ساخت لیست جدید از eventها بر اساس اندیس‌های نگه داشته شده
    # این کار را مستقیماً روی session_events انجام می‌دهیم تا سادگی حفظ شود
    
    deduplicated_events = []
    if not session_events: return []
    
    deduplicated_events.append(session_events[0])
    for i in range(1, len(session_events)):
        # مقایسه آیتم فعلی با آیتم قبلی در لیست *جدید* (deduplicated_events)
        # برای تطابق کامل با delete_dups کد نمونه، باید روی لیست اصلی کار کنیم
        # و سپس بر اساس to_keep_indices بازسازی کنیم.
        # ساده‌تر:
        if session_events[i][0] != session_events[i-1][0]: # اگر آیتم فعلی با آیتم قبلی *در لیست اصلی* متفاوت بود
             deduplicated_events.append(session_events[i])
    
    # کد نمونه شما این کار را بعد از نگاشت انجام می‌دهد، اما می‌توانیم اینجا هم انجام دهیم
    # برای سادگی، اجازه دهید فعلاً این تابع را نگه داریم و بعداً در صورت نیاز دقیق‌تر پیاده‌سازی کنیم
    # یا اینکه ساختار داده را به سه لیست جداگانه از ابتدا تغییر دهیم.
    # فعلاً، فرض می‌کنیم آیتم‌های متوالی تکراری در داده خام Yoochoose نادر هستند یا اهمیت زیادی ندارند
    # و این بخش را ساده نگه می‌داریم. کد نمونه شما پس از نگاشت به ID این کار را می‌کند.
    # برای تطابق، این تابع را بعد از نگاشت آیتم‌ها فراخوانی می‌کنیم.
    return session_events # فعلا بدون تغییر برمی‌گردانیم، بعد از نگاشت اعمال می‌شود

item_to_id_map = {}
current_item_id = 1 # شروع از ۱

# پردازش و نگاشت مجموعه آموزشی
train_sequences_raw_ts = [] # لیستی از لیست‌های [(item_id_mapped, raw_timestamp_float)]
train_session_actual_items = [] # لیستی از لیست‌های [item_id_mapped]
train_session_timestamps_abs = [] # لیستی از لیست‌های [raw_timestamp_float]


print("Processing training data...")
for s_id, _ in tqdm(train_session_ids_dates):
    session_events = sess_clicks_ts_cat[s_id] # [(item_id_orig, category, timestamp_float), ...]
    
    # نگاشت آیتم‌ها و استخراج timestamp خام
    current_session_mapped_items = []
    current_session_abs_timestamps = []

    for item_orig, _, ts_float in session_events:
        if item_orig not in item_to_id_map:
            item_to_id_map[item_orig] = current_item_id
            current_item_id += 1
        current_session_mapped_items.append(item_to_id_map[item_orig])
        current_session_abs_timestamps.append(ts_float)
        
    # حذف آیتم‌های متوالی تکراری (پس از نگاشت)
    # برای این کار نیاز به یک تابع delete_dups مشابه کد نمونه داریم که روی لیست‌های نگاشت شده کار کند
    if len(current_session_mapped_items) >= MIN_SESSION_LENGTH : # کد نمونه هم این شرط را پس از نگاشت دارد
        # تابع delete_dups ساده شده برای دو لیست
        # (کد نمونه category را هم داشت، اینجا فعلا فقط آیتم و تایم‌استمپ)
        if current_session_mapped_items: # اطمینان از خالی نبودن
            dedup_items = [current_session_mapped_items[0]]
            dedup_ts = [current_session_abs_timestamps[0]]
            for i in range(1, len(current_session_mapped_items)):
                if current_session_mapped_items[i] != dedup_items[-1]: # مقایسه با آخرین آیتم اضافه شده به لیست dedup
                    dedup_items.append(current_session_mapped_items[i])
                    dedup_ts.append(current_session_abs_timestamps[i])
            
            if len(dedup_items) >= MIN_SESSION_LENGTH:
                train_session_actual_items.append(dedup_items)
                train_session_timestamps_abs.append(dedup_ts)

print(f"Total items after training processing (item_ctr): {current_item_id}")

# پردازش و نگاشت مجموعه آزمون
test_session_actual_items = []
test_session_timestamps_abs = []

print("Processing test data...")
for s_id, _ in tqdm(test_session_ids_dates):
    session_events = sess_clicks_ts_cat[s_id]
    
    current_session_mapped_items = []
    current_session_abs_timestamps = []

    for item_orig, _, ts_float in session_events:
        if item_orig in item_to_id_map: # فقط آیتم‌هایی که در آموزش بوده‌اند
            current_session_mapped_items.append(item_to_id_map[item_orig])
            current_session_abs_timestamps.append(ts_float)
            
    if len(current_session_mapped_items) >= MIN_SESSION_LENGTH:
        if current_session_mapped_items: # اطمینان از خالی نبودن
            dedup_items = [current_session_mapped_items[0]]
            dedup_ts = [current_session_abs_timestamps[0]]
            for i in range(1, len(current_session_mapped_items)):
                if current_session_mapped_items[i] != dedup_items[-1]:
                    dedup_items.append(current_session_mapped_items[i])
                    dedup_ts.append(current_session_abs_timestamps[i])

            if len(dedup_items) >= MIN_SESSION_LENGTH:
                test_session_actual_items.append(dedup_items)
                test_session_timestamps_abs.append(dedup_ts)


# تابع process_seqs برای تقسیم به ورودی و هدف و محاسبه اختلاف زمانی
def process_sequences_and_timediffs(session_item_list, session_ts_abs_list):
    out_sequences = []  # ورودی مدل: seq[:-1]
    out_targets = []    # هدف مدل: seq[-1] (در کد شما) یا seq[1:] (در کد نمونه)
                        # برای سازگاری با مدل شما، هدف را seq[-1] در نظر می‌گیریم
    out_time_diffs = [] # اختلاف زمانی

    for items, timestamps_abs in zip(tqdm(session_item_list), session_ts_abs_list):
        if not items: continue

        # محاسبه اختلاف زمانی
        current_time_diffs = [0.0] # اولین آیتم اختلاف زمانی صفر دارد
        for i in range(len(timestamps_abs) - 1):
            diff = timestamps_abs[i+1] - timestamps_abs[i]
            current_time_diffs.append(diff if diff >= 0 else 0.0) # اختلاف منفی را صفر می‌کنیم
            
        # سازگاری با مدل شما: ورودی شامل همه آیتم‌ها، هدف آخرین آیتم
        out_sequences.append(list(items)) # لیست کامل آیتم‌ها به عنوان ورودی
        out_targets.append(items[-1])     # آخرین آیتم به عنوان هدف
        out_time_diffs.append(current_time_diffs)
        
        # اگر بخواهیم مشابه کد نمونه عمل کنیم (process_seqs در کد نمونه):
        # out_sequences.append(list(items[:-1])) # ورودی: همه به جز آخر
        # out_targets.append(items[-1]) # هدف: فقط آخرین آیتم (برای سازگاری با کد نمونه که labs += [tar] و tar = seq[1:] دارد، یعنی labs شامل لیست اهداف است)
                                     # اما مدل شما یک هدف واحد برای هر سکانس انتظار دارد
        # out_time_diffs.append(current_time_diffs[:-1] if current_time_diffs else [])


    return out_sequences, out_targets, out_time_diffs


print("Final processing for train sequences and time diffs...")
train_sequences, train_targets, train_time_diffs = process_sequences_and_timediffs(
    train_session_actual_items, train_session_timestamps_abs
)

print("Final processing for test sequences and time diffs...")
test_sequences, test_targets, test_time_diffs = process_sequences_and_timediffs(
    test_session_actual_items, test_session_timestamps_abs
)

print(f"Final train sequences: {len(train_sequences)}")
print(f"Final test sequences: {len(test_sequences)}")
print(f"Number of unique items in mapping: {len(item_to_id_map)}")


# ذخیره داده‌ها در فرمت مورد انتظار مدل شما
# train.txt: (train_sequences, train_targets) -> train_targets لیستی از آخرین آیتم هر سکانس است
# test.txt: (test_sequences, test_targets)
# time_data.pkl: {'train_time_diffs': train_time_diffs, 'test_time_diffs': test_time_diffs}

output_train_file = os.path.join(opt.output_dir, 'train.txt')
output_test_file = os.path.join(opt.output_dir, 'test.txt')
output_time_data_file = os.path.join(opt.output_dir, 'time_data.pkl')

with open(output_train_file, 'wb') as f:
    pickle.dump((train_sequences, train_targets), f)
print(f"Saved training data to {output_train_file}")

with open(output_test_file, 'wb') as f:
    pickle.dump((test_sequences, test_targets), f)
print(f"Saved test data to {output_test_file}")

with open(output_time_data_file, 'wb') as f:
    pickle.dump({
        'train_time_diffs': train_time_diffs,
        'test_time_diffs': test_time_diffs
    }, f)
print(f"Saved time data to {output_time_data_file}")

print("-- Data saved @ %s" % datetime.datetime.now())
print("Done.")