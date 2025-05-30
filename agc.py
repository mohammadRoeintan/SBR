#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018
Modified in June, 2024 to include time differences
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = ''
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose-clicks.dat'

print("-- Starting @ %ss" % datetime.datetime.now())

# ساختارهای داده‌ای جدید برای ذخیره‌سازی تایم‌استمپ‌ها
sess_timestamps = {}  # {session_id: [timestamp1, timestamp2, ...]}

with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    
    for data in reader:
        sessid = data['session_id']
        if opt.dataset == 'yoochoose':
            item = data['item_id']
            timestamp = data['timestamp']
        else:
            item = data['item_id'], int(data['timeframe'])
            timestamp = data['eventdate']

        # ذخیره تایم‌استمپ
        if sessid in sess_timestamps:
            sess_timestamps[sessid].append(timestamp)
        else:
            sess_timestamps[sessid] = [timestamp]
        
        # بقیه کد مثل قبل...
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        
        curid = sessid
        curdate = timestamp

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        
        ctr += 1
    
    # پردازش نهایی
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
    
    sess_date[curid] = date

print("-- Reading data @ %ss" % datetime.datetime.now())

# فیلتر سشن‌های طول 1
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]
        if s in sess_timestamps:
            del sess_timestamps[s]

# شمارش آیتم‌ها و فیلتر
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
    if len(filseq) < 2:
        del sess_clicks[s]
        del sess_date[s]
        if s in sess_timestamps:
            del sess_timestamps[s]
    else:
        sess_clicks[s] = filseq

# تقسیم داده‌ها
dates = list(sess_date.items())
maxdate = max(date for _, date in dates)

# 7 روز برای تست
splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)
tra_sess = [(s, date) for s, date in dates if date < splitdate]
tes_sess = [(s, date) for s, date in dates if date > splitdate]

# مرتب‌سازی سشن‌ها
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))
print(len(tra_sess))
print(len(tes_sess))
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# تابع برای ایجاد داده‌های آموزشی
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    train_times = []  # لیست جدید برای اختلاف‌های زمانی
    item_ctr = 1
    item_dict = {}
    
    for s, date in tra_sess:
        seq = sess_clicks[s]
        timestamps = sess_timestamps.get(s, [])
        outseq = []
        time_diffs = [0]  # اولین رویداد اختلاف زمانی 0 دارد
        
        # محاسبه اختلاف زمانی بین رویدادها
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                try:
                    if opt.dataset == 'yoochoose':
                        t1 = datetime.datetime.strptime(timestamps[i-1][:19], '%Y-%m-%dT%H:%M:%S')
                        t2 = datetime.datetime.strptime(timestamps[i][:19], '%Y-%m-%dT%H:%M:%S')
                    else:
                        t1 = datetime.datetime.strptime(timestamps[i-1], '%Y-%m-%d')
                        t2 = datetime.datetime.strptime(timestamps[i], '%Y-%m-%d')
                    diff = (t2 - t1).total_seconds()
                    time_diffs.append(diff)
                except:
                    time_diffs.append(0)
        
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        
        if len(outseq) < 2:
            continue
        
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
        train_times += [time_diffs]  # ذخیره اختلاف‌های زمانی
    
    print(item_ctr)
    return train_ids, train_dates, train_seqs, train_times, item_dict

# تابع برای ایجاد داده‌های تست
def obtian_tes(item_dict):
    test_ids = []
    test_seqs = []
    test_dates = []
    test_times = []  # لیست جدید برای اختلاف‌های زمانی
    
    for s, date in tes_sess:
        seq = sess_clicks[s]
        timestamps = sess_timestamps.get(s, [])
        outseq = []
        time_diffs = [0]  # اولین رویداد اختلاف زمانی 0 دارد
        
        # محاسبه اختلاف زمانی بین رویدادها
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                try:
                    if opt.dataset == 'yoochoose':
                        t1 = datetime.datetime.strptime(timestamps[i-1][:19], '%Y-%m-%dT%H:%M:%S')
                        t2 = datetime.datetime.strptime(timestamps[i][:19], '%Y-%m-%dT%H:%M:%S')
                    else:
                        t1 = datetime.datetime.strptime(timestamps[i-1], '%Y-%m-%d')
                        t2 = datetime.datetime.strptime(timestamps[i], '%Y-%m-%d')
                    diff = (t2 - t1).total_seconds()
                    time_diffs.append(diff)
                except:
                    time_diffs.append(0)
        
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        
        if len(outseq) < 2:
            continue
        
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
        test_times += [time_diffs]  # ذخیره اختلاف‌های زمانی
    
    return test_ids, test_dates, test_seqs, test_times

# ایجاد داده‌های آموزشی و تست
tra_ids, tra_dates, tra_seqs, tra_times, item_dict = obtian_tra()
tes_ids, tes_dates, tes_seqs, tes_times = obtian_tes(item_dict)

# پردازش سکانس‌ها
def process_seqs(iseqs, idates, itimes):
    out_seqs = []
    out_dates = []
    out_times = []  # لیست جدید برای زمان‌ها
    labs = []
    ids = []
    
    for id, (seq, date, times) in enumerate(zip(iseqs, idates, itimes)):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            out_times += [times[:len(seq)-i]]  # ذخیره زمان‌ها برای این سکانس
            ids += [id]
    
    return out_seqs, out_dates, labs, ids, out_times

# پردازش نهایی
tr_seqs, tr_dates, tr_labs, tr_ids, tr_times = process_seqs(tra_seqs, tra_dates, tra_times)
te_seqs, te_dates, te_labs, te_ids, te_times = process_seqs(tes_seqs, tes_dates, tes_times)

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

# ذخیره داده‌های زمانی
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    
    with open('diginetica/time_data.pkl', 'wb') as f:
        pickle.dump(tr_times + te_times, f)
    
    # بقیه ذخیره‌سازی مانند قبل...

elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    
    # ذخیره داده‌های زمانی برای yoochoose1_64
    with open('yoochoose1_64/time_data.pkl', 'wb') as f:
        pickle.dump(tr_times, f)
    
    # بقیه ذخیره‌سازی مانند قبل...

print('Done. Time differences data saved to time_data.pkl')
