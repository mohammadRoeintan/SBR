#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018
Modified in June, 2024 for Yoochoose dataset
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: yoochoose')
opt = parser.parse_args()
print(opt)

# تنظیمات خاص دیتاست Yoochoose
if opt.dataset == 'yoochoose':
    dataset = '/kaggle/input/yoochoose/yoochoose-clicks.dat'
    # تعریف نام ستون‌ها بر اساس مستندات
    COLUMNS = ['session_id', 'timestamp', 'item_id', 'category']
    
print("-- Starting @ %ss" % datetime.datetime.now())

# ساختارهای داده‌ای
sess_clicks = defaultdict(list)      # {session_id: [item1, item2, ...]}
sess_timestamps = defaultdict(list)  # {session_id: [timestamp1, timestamp2, ...]}
sess_last_date = {}                  # {session_id: last_timestamp}
item_counts = defaultdict(int)       # {item_id: count}

# خواندن فایل دیتاست
with open(dataset, "r") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        # بررسی تعداد ستون‌ها
        if len(row) != 4:
            continue
            
        # استخراج داده‌ها بر اساس موقعیت ستون
        session_id = row[0].strip()
        timestamp = row[1].strip()
        item_id = row[2].strip()
        category = row[3].strip()  # استفاده نشده اما برای کامل بودن استخراج می‌شود
        
        # ذخیره داده‌ها
        sess_clicks[session_id].append(item_id)
        sess_timestamps[session_id].append(timestamp)
        sess_last_date[session_id] = timestamp
        item_counts[item_id] += 1

print(f"Total sessions: {len(sess_clicks)}")
print(f"Total items: {len(item_counts)}")
print("-- Reading data @ %ss" % datetime.datetime.now())

# فیلتر سشن‌های طول 1 و آیتم‌های نادر
MIN_SESSION_LENGTH = 2
MIN_ITEM_COUNT = 5

# حذف سشن‌های کوتاه
for session_id in list(sess_clicks.keys()):
    if len(sess_clicks[session_id]) < MIN_SESSION_LENGTH:
        del sess_clicks[session_id]
        del sess_timestamps[session_id]
        if session_id in sess_last_date:
            del sess_last_date[session_id]

# فیلتر آیتم‌های نادر
for session_id in list(sess_clicks.keys()):
    filtered_items = []
    filtered_timestamps = []
    for item, timestamp in zip(sess_clicks[session_id], sess_timestamps[session_id]):
        if item_counts[item] >= MIN_ITEM_COUNT:
            filtered_items.append(item)
            filtered_timestamps.append(timestamp)
    
    if len(filtered_items) >= MIN_SESSION_LENGTH:
        sess_clicks[session_id] = filtered_items
        sess_timestamps[session_id] = filtered_timestamps
    else:
        del sess_clicks[session_id]
        del sess_timestamps[session_id]
        if session_id in sess_last_date:
            del sess_last_date[session_id]

print(f"Sessions after filtering: {len(sess_clicks)}")

# تبدیل تایم‌استمپ‌ها به ثانیه از epoch
def convert_yoochoose_timestamp(timestamp_str):
    try:
        # فرمت: YYYY-MM-DDThh:mm:ss.SSSZ
        dt = datetime.datetime.strptime(timestamp_str[:19], '%Y-%m-%dT%H:%M:%S')
        return time.mktime(dt.timetuple())
    except:
        return 0

# محاسبه زمان هر سشن (آخرین تایم‌استمپ)
sess_times = {}
for session_id, timestamps in sess_timestamps.items():
    if timestamps:
        last_timestamp = timestamps[-1]
        sess_times[session_id] = convert_yoochoose_timestamp(last_timestamp)

# تقسیم داده به train/test بر اساس زمان
all_sessions = list(sess_times.items())
all_sessions.sort(key=lambda x: x[1])  # مرتب‌سازی بر اساس زمان

# پیدا کردن زمان تقسیم (7 روز قبل از آخرین رویداد)
if all_sessions:
    max_time = all_sessions[-1][1]
    split_time = max_time - (7 * 24 * 3600)  # 7 روز قبل
else:
    split_time = 0

train_sessions = [s for s in all_sessions if s[1] < split_time]
test_sessions = [s for s in all_sessions if s[1] >= split_time]

print(f"Train sessions: {len(train_sessions)}")
print(f"Test sessions: {len(test_sessions)}")

# ایجاد دیکشنری برای نگاشت آیتم‌ها به ID
item_to_id = {}
current_id = 1

def map_items(session_items):
    global current_id
    mapped = []
    for item in session_items:
        if item not in item_to_id:
            item_to_id[item] = current_id
            current_id += 1
        mapped.append(item_to_id[item])
    return mapped

# پردازش سشن‌های آموزشی
train_sequences = []
train_time_diffs = []

for session_id, _ in train_sessions:
    items = sess_clicks[session_id]
    timestamps = sess_timestamps[session_id]
    
    # محاسبه اختلاف زمانی
    time_diffs = [0]  # اولین آیتم اختلاف زمانی 0 دارد
    if len(timestamps) > 1:
        prev_time = convert_yoochoose_timestamp(timestamps[0])
        for ts in timestamps[1:]:
            current_time = convert_yoochoose_timestamp(ts)
            time_diffs.append(current_time - prev_time)
            prev_time = current_time
    
    # نگاشت آیتم‌ها به ID
    mapped_items = map_items(items)
    train_sequences.append(mapped_items)
    train_time_diffs.append(time_diffs)

# پردازش سشن‌های تست
test_sequences = []
test_time_diffs = []

for session_id, _ in test_sessions:
    items = sess_clicks[session_id]
    timestamps = sess_timestamps[session_id]
    
    # محاسبه اختلاف زمانی
    time_diffs = [0]
    if len(timestamps) > 1:
        prev_time = convert_yoochoose_timestamp(timestamps[0])
        for ts in timestamps[1:]:
            current_time = convert_yoochoose_timestamp(ts)
            time_diffs.append(current_time - prev_time)
            prev_time = current_time
    
    # نگاشت فقط آیتم‌هایی که در train دیده شده‌اند
    mapped_items = [item_to_id[item] for item in items if item in item_to_id]
    if len(mapped_items) >= MIN_SESSION_LENGTH:
        test_sequences.append(mapped_items)
        test_time_diffs.append(time_diffs)

print(f"Final item count: {len(item_to_id)}")
print(f"Train sequences: {len(train_sequences)}")
print(f"Test sequences: {len(test_sequences)}")

# ذخیره داده‌ها
if not os.path.exists('yoochoose1_64'):
    os.makedirs('yoochoose1_64')

# ذخیره train/test
with open('yoochoose1_64/train.txt', 'wb') as f:
    pickle.dump((train_sequences, [seq[-1] for seq in train_sequences]), f)

with open('yoochoose1_64/test.txt', 'wb') as f:
    pickle.dump((test_sequences, [seq[-1] for seq in test_sequences]), f)

# ذخیره داده‌های زمانی
with open('yoochoose1_64/time_data.pkl', 'wb') as f:
    pickle.dump({
        'train_time_diffs': train_time_diffs,
        'test_time_diffs': test_time_diffs
    }, f)

print("-- Data saved @ %ss" % datetime.datetime.now())
print("Done.")
