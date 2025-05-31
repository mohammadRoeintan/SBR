##################################################################
# This code was adapted from https://github.com/CRIPAC-DIG/TAGNN #
# and STAR: https://github.com/yeganegi-reza/STAR                #
##################################################################
##################################################################
# This code was adapted from https://github.com/CRIPAC-DIG/TAGNN #
# and STAR: https://github.com/yeganegi-reza/STAR                #
##################################################################

import numpy as np
import random
from collections import defaultdict

def split_validation(data, valid_portion=0.1):
    # داده‌ها را به دو بخش train و valid تقسیم می‌کند
    train_set_x = data[0]
    train_set_y = data[1]
    
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)  # داده‌ها را تصادفی می‌کنیم
    
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    
    # ایجاد داده‌های آموزشی و اعتبارسنجی
    train_x = [train_set_x[i] for i in sidx[:n_train]]
    train_y = [train_set_y[i] for i in sidx[:n_train]]
    
    valid_x = [train_set_x[i] for i in sidx[n_train:]]
    valid_y = [train_set_y[i] for i in sidx[n_train:]]
    
    return (train_x, train_y), (valid_x, valid_y)

def data_masks(all_usr_pois, item_tail, max_len):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = min(max(us_lens), max_len) if max_len > 0 else max(us_lens)
    if not us_lens: # اگر لیست خالی باشد
        len_max = max_len if max_len > 0 else 0

    us_pois = []
    for upois in all_usr_pois:
        current_len = len(upois)
        padding_len = len_max - min(current_len, len_max)
        us_pois.append(upois[:min(current_len, len_max)] + item_tail * padding_len)

    us_msks = [[1] * min(le, len_max) + [0] * (len_max - min(le, len_max)) for le in us_lens]
    
    us_pos = [
        list(range(1, min(le, len_max) + 1)) + [0] * (len_max - min(le, len_max))
        for le in us_lens
    ]
    
    return us_pois, us_msks, us_pos, len_max

class Dataset():
    def __init__(self, data, time_data=None, shuffle=False, opt=None):
        # تبدیل داده‌ها به لیست
        self.raw_inputs = [list(seq) for seq in data[0]]
        
        # تبدیل اهداف به لیست اگر آرایه NumPy باشد
        if isinstance(data[1], np.ndarray):
            self.targets = data[1].tolist()
        else:
            self.targets = data[1]
            
        # اگر time_data ارائه نشده باشد، آن را با صفر پر می‌کنیم.
        # در غیر این صورت، اطمینان حاصل می‌کنیم که یک لیست از لیست‌ها است.
        if time_data is None:
            self.time_data_raw = [[] for _ in self.raw_inputs]
        else:
            self.time_data_raw = [list(td) if not isinstance(td, list) else td for td in time_data]

        self.length = len(self.raw_inputs)
        self.shuffle = shuffle
        self.opt = opt
        
        if opt and hasattr(opt, 'max_len') and opt.max_len > 0:
            self.len_max = opt.max_len
        else:
            us_lens_init = [len(upois) for upois in self.raw_inputs]
            self.len_max = max(us_lens_init) if us_lens_init else 0
        
        padded_inputs, padded_mask, padded_pos, _ = data_masks(self.raw_inputs, [0], self.len_max)
        self.inputs = np.asarray(padded_inputs)
        self.mask = np.asarray(padded_mask)
        self.positions = np.asarray(padded_pos)
        
        # محاسبه و پد کردن تفاوت‌های زمانی با استفاده از متد اصلاح شده
        self.time_diffs_padded = self._pad_and_prepare_time_diffs()

    def _pad_and_prepare_time_diffs(self):
        """
        از تفاوت‌های زمانی خام (از agc.py) استفاده کرده و آن‌ها را به self.len_max پد می‌کند.
        هر لیست تفاوت زمانی در self.time_data_raw باید با طول توالی متناظرش در self.raw_inputs همخوانی داشته باشد
        و معمولاً با یک 0 برای آیتم اول شروع می‌شود.
        """
        padded_time_diffs_matrix = []
        for i, session_raw_input in enumerate(self.raw_inputs):
            # اطمینان از وجود داده زمانی برای سشن فعلی
            if i < len(self.time_data_raw):
                current_session_time_diffs = list(self.time_data_raw[i]) # کپی برای جلوگیری از تغییر داده اصلی
            else:
                # اگر داده زمانی موجود نباشد، یک لیست خالی یا پر از صفر ایجاد کن
                # طول آن باید با طول session_raw_input همخوانی داشته باشد
                current_session_time_diffs = [0.0] * len(session_raw_input)

            # طول داده زمانی باید با طول توالی آیتم‌ها یکی باشد
            # (agc.py این را تضمین می‌کند چون یک 0 برای اولین آیتم اضافه می‌کند)
            if len(current_session_time_diffs) != len(session_raw_input) and len(session_raw_input) > 0 :
                # این حالت نباید رخ دهد اگر agc.py درست کار کند.
                # اگر رخ داد، برای جلوگیری از خطا، آن را با صفر پر می‌کنیم یا کوتاه می‌کنیم.
                # اما بهتر است یک هشدار چاپ شود.
                # print(f"Warning: Mismatch in length of items ({len(session_raw_input)}) and time_diffs ({len(current_session_time_diffs)}) for session index {i}. Adjusting time_diffs.")
                if len(current_session_time_diffs) < len(session_raw_input):
                    current_session_time_diffs.extend([0.0] * (len(session_raw_input) - len(current_session_time_diffs)))
                else:
                    current_session_time_diffs = current_session_time_diffs[:len(session_raw_input)]
            
            # پد کردن/کوتاه کردن لیست تفاوت‌های زمانی به self.len_max
            # این همان طول padded_inputs است.
            if len(current_session_time_diffs) < self.len_max:
                current_session_time_diffs.extend([0.0] * (self.len_max - len(current_session_time_diffs)))
            elif len(current_session_time_diffs) > self.len_max:
                current_session_time_diffs = current_session_time_diffs[:self.len_max]
            
            padded_time_diffs_matrix.append(current_session_time_diffs)
            
        return np.array(padded_time_diffs_matrix, dtype=np.float32)
        
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            
            shuffled_indices = shuffled_arg.tolist()
            
            self.raw_inputs = [self.raw_inputs[i] for i in shuffled_indices]
            self.targets = [self.targets[i] for i in shuffled_indices]
            # self.time_diffs_padded هم باید متناسب با این shuffle بازآرایی شود
            # اما چون self.time_data_raw بازآرایی می‌شود و self.time_diffs_padded
            # در get_slice با استفاده از ایندکس‌های بچ ساخته می‌شود، نیاز به بازآرایی مستقیم نیست
            # مگر اینکه get_slice مستقیما از self.time_diffs_padded استفاده کند.
            # با توجه به ساختار get_slice که از self.time_diffs_padded استفاده می‌کند،
            # باید self.time_data_raw و سپس self.time_diffs_padded را هم شافل کنیم.
            self.time_data_raw = [self.time_data_raw[i] for i in shuffled_indices]


            # بعد از شافل کردن داده‌های خام، نیاز است که داده‌های پد شده هم بروز شوند
            current_padded_inputs, current_padded_mask, current_padded_pos, _ = data_masks(self.raw_inputs, [0], self.len_max)
            self.inputs = np.asarray(current_padded_inputs)
            self.mask = np.asarray(current_padded_mask)
            self.positions = np.asarray(current_padded_pos)
            self.time_diffs_padded = self._pad_and_prepare_time_diffs() # بازسازی با داده‌های شافل شده

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        
        slices = []
        for i in range(n_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.length)
            if start_idx < end_idx : # اطمینان از اینکه اسلایس خالی نیست
                 slices.append(np.arange(start_idx, end_idx))
        
        return slices

    def _augment_sequence_item_dropout(self, seq_items_padded, seq_time_diffs_padded, drop_prob):
        # seq_items_padded و seq_time_diffs_padded باید هم‌طول باشند (self.len_max)
        
        augmented_seq_items = []
        augmented_time_diffs = []
        
        # همیشه حداقل یک آیتم را نگه می‌داریم اگر drop_prob برابر 1.0 نباشد
        # و توالی اصلی خالی نباشد.
        non_pad_items_indices = [i for i, item in enumerate(seq_items_padded) if item != 0]

        if not non_pad_items_indices: # اگر توالی فقط شامل پدینگ باشد
            return list(seq_items_padded), list(seq_time_diffs_padded)

        # حداقل یک آیتم نگه داشته شود (اولین آیتم غیر پد)
        # این بخش برای اطمینان از عدم ایجاد توالی کاملا خالی است اگر drop_prob بالا باشد
        # و به خصوص اگر بخواهیم از اولین آیتم برای SSL استفاده کنیم.
        # اما منطق فعلی TAGNN/STAR ممکن است این را نیاز نداشته باشد.
        # در اینجا فرض می‌کنیم که اگر پس از دراپ‌آوت چیزی نماند، اشکالی ندارد و بعدا مدیریت می‌شود.

        for i in range(len(seq_items_padded)):
            item = seq_items_padded[i]
            time_d = seq_time_diffs_padded[i]
            
            if item == 0: # اگر آیتم پدینگ است، آن را نگه دار
                augmented_seq_items.append(item)
                augmented_time_diffs.append(time_d) # زمان پدینگ هم معمولا 0 است
                continue

            if random.random() > drop_prob:
                augmented_seq_items.append(item)
                augmented_time_diffs.append(time_d)
            # اگر آیتم دراپ شد، دیگر به لیست اضافه نمی‌شود
            # و چون طول ورودی‌ها ثابت است (self.len_max)،
            # دراپ کردن آیتم‌ها باعث کوتاه شدن لیست نمی‌شود، بلکه جای آن‌ها خالی می‌ماند.
            # برای اینکه شبیه TAGNN عمل کنیم، باید آیتم‌های دراپ شده را با 0 (پدینگ) جایگزین کنیم.

        # اگر پس از دراپ کردن، توالی خالی شد (همه آیتم‌های غیر پد دراپ شدند)
        # برای جلوگیری از خطا، حداقل اولین آیتم اصلی را برمی‌گردانیم (اگر توالی اصلی آیتم داشت)
        # یا یک توالی پد شده برمی‌گردانیم.
        # این بخش برای robust بودن است.
        # یک راه ساده‌تر این است که مطمئن شویم حداقل یک آیتم باقی می‌ماند اگر توالی اصلی خالی نبود.
        
        # اصلاح: اگر قرار است آیتم‌ها حذف شوند و سپس با پدینگ پر شوند، منطق متفاوت است.
        # منطق فعلی کد اصلی TAGNN/STAR برای get_slice به نظر می‌رسد که ابتدا توالی‌های خام را augment می‌کند
        # و سپس آن‌ها را پد می‌کند. اینجا ما روی توالی‌های از قبل پد شده کار می‌کنیم.
        # برای سادگی، فرض می‌کنیم که اگر آیتمی دراپ شد، با 0 جایگزین نمی‌شود،
        # بلکه فقط آیتم‌های نگه داشته شده جمع‌آوری می‌شوند و سپس کل توالی به self.len_max پد می‌شود.
        # اما چون ورودی‌ها از قبل پد شده‌اند، بهتر است آیتم‌های دراپ شده را با 0 جایگزین کنیم
        # تا طول حفظ شود.
        
        final_augmented_seq_items = []
        final_augmented_time_diffs = []
        
        # اگر قرار است توالی اصلی دست نخورده بماند و فقط آیتم‌ها برای ساخت view جدید دراپ شوند
        # و سپس view جدید پد شود، باید از raw_inputs کار را شروع کرد.
        # متد فعلی روی داده‌های از قبل پد شده از self.inputs و self.time_diffs_padded کار می‌کند.
        # برای SSL معمولا یک view از داده اصلی ساخته می‌شود.
        
        # بازنویسی برای حفظ طول و جایگزینی آیتم‌های دراپ شده با پدینگ (0)
        # این با فرض این است که drop_prob برای آیتم‌های غیر پد اعمال می‌شود.
        
        temp_augmented_seq = []
        temp_augmented_time = []
        
        original_length_before_pad = 0
        for item_val in seq_items_padded:
            if item_val != 0:
                original_length_before_pad +=1
        
        # اگر توالی اصلی (قبل از پدینگ) خالی بود، همان را برگردان
        if original_length_before_pad == 0:
            return list(seq_items_padded), list(seq_time_diffs_padded)

        items_kept_count = 0
        for i in range(self.len_max):
            item = seq_items_padded[i]
            time_d = seq_time_diffs_padded[i]

            if item != 0: # فقط روی آیتم‌های غیر پدینگ دراپ اعمال کن
                if random.random() > drop_prob:
                    temp_augmented_seq.append(item)
                    temp_augmented_time.append(time_d)
                    items_kept_count+=1
            # آیتم‌های پدینگ در این مرحله اضافه نمی‌شوند.
            
        # اگر هیچ آیتمی نگه داشته نشد، و توالی اصلی آیتم داشت، اولین آیتم اصلی را نگه دار
        if items_kept_count == 0 and original_length_before_pad > 0:
             first_non_pad_idx = -1
             for idx, item_val in enumerate(seq_items_padded):
                 if item_val != 0:
                     first_non_pad_idx = idx
                     break
             if first_non_pad_idx != -1:
                 temp_augmented_seq = [seq_items_padded[first_non_pad_idx]]
                 temp_augmented_time = [seq_time_diffs_padded[first_non_pad_idx]]


        # حالا temp_augmented_seq و temp_augmented_time را به self.len_max پد کن
        padded_augmented_seq = temp_augmented_seq + [0] * (self.len_max - len(temp_augmented_seq))
        padded_augmented_time = temp_augmented_time + [0.0] * (self.len_max - len(temp_augmented_time))
        
        return padded_augmented_seq, padded_augmented_time


    def _get_graph_data_for_view(self, current_inputs_batch_padded_items, current_time_diffs_batch_padded):
        # current_inputs_batch_padded_items: لیستی از توالی‌های آیتم پد شده (هر توالی یک لیست است)
        # current_time_diffs_batch_padded: لیستی از توالی‌های تفاوت زمانی پد شده
        
        batch_size = len(current_inputs_batch_padded_items)
        items_list_unique_nodes, A_list, alias_inputs_list = [], [], []
        mask_list_for_view = [] 
        positions_list_for_view = []
        
        # تعیین بیشترین تعداد گره یکتا در بچ برای پد کردن ماتریس‌ها و لیست گره‌ها
        batch_max_n_node = 0
        temp_n_nodes_for_batch = []
        for u_input_single_items_padded in current_inputs_batch_padded_items:
            unique_nodes_in_seq = np.unique(np.array(u_input_single_items_padded))
            unique_nodes_in_seq = unique_nodes_in_seq[unique_nodes_in_seq != 0] # حذف پدینگ (0)
            num_unique = len(unique_nodes_in_seq)
            temp_n_nodes_for_batch.append(num_unique if num_unique > 0 else 1) # حداقل یک گره برای جلوگیری از خطای تقسیم بر صفر
        
        if temp_n_nodes_for_batch: # اگر بچ خالی نباشد
            batch_max_n_node = np.max(temp_n_nodes_for_batch)
        else: # اگر بچ خالی است
            batch_max_n_node = 1 # یا هر مقدار پیش‌فرض مناسب دیگر، مثلا self.len_max اگرچه منطقی نیست

        for idx, u_input_single_items_padded in enumerate(current_inputs_batch_padded_items):
            # ساخت ماسک و موقعیت بر اساس آیتم‌های پد شده فعلی (که ممکن است augment شده باشند)
            current_mask = [1 if item != 0 else 0 for item in u_input_single_items_padded]
            mask_list_for_view.append(current_mask)
            
            current_pos_seq = []
            effective_len = 0
            for item_val in u_input_single_items_padded:
                if item_val != 0:
                    effective_len += 1
                    current_pos_seq.append(effective_len)
                else:
                    current_pos_seq.append(0)
            positions_list_for_view.append(current_pos_seq)

            # گره‌های یکتا برای ساخت گراف (بدون پدینگ)
            node_unique_for_graph = np.unique(np.array(u_input_single_items_padded))
            node_unique_for_graph = node_unique_for_graph[node_unique_for_graph != 0]
            
            if len(node_unique_for_graph) == 0: # اگر پس از augment، توالی خالی شد
                # یک گره ساختگی (مثلا 0) اضافه می‌کنیم تا از خطا جلوگیری شود.
                # مدل باید بتواند با ورودی‌های خالی یا پد شده کار کند.
                # در اینجا یک گره 0 اضافه می‌کنیم و ماتریس همجواری صفر خواهد بود.
                # این باید با نحوه مدیریت padding_idx در مدل هماهنگ باشد.
                # اگر padding_idx=0 است، items_list_unique_nodes باید شامل 0 برای پدینگ باشد.
                # اما معمولا گره‌های گراف شامل padding_idx نمی‌شوند.
                # برای سادگی، اگر گره یکتایی وجود ندارد، یک لیست خالی یا با یک 0 (به عنوان پدینگ گره) برمیگردانیم.
                 items_list_unique_nodes.append([0] * batch_max_n_node) # لیست گره‌های پد شده
                 A_list.append(np.zeros((batch_max_n_node, batch_max_n_node * 2))) # ماتریس همجواری صفر پد شده
                 alias_inputs_list.append([0] * self.len_max) # آلیاس‌های پد شده
                 continue # به سشن بعدی برو


            # پد کردن لیست گره‌های یکتا
            padded_unique_nodes = node_unique_for_graph.tolist() + [0] * (batch_max_n_node - len(node_unique_for_graph))
            items_list_unique_nodes.append(padded_unique_nodes)
            
            # ساخت ماتریس همجواری
            u_A = np.zeros((batch_max_n_node, batch_max_n_node))
            # نگاشت از item_id به ایندکس در node_unique_for_graph (0 تا len(node_unique_for_graph)-1)
            item_to_idx_map = {item_id: k for k, item_id in enumerate(node_unique_for_graph)}

            # فقط از آیتم‌های غیر پد شده در توالی برای ساخت یال‌ها استفاده کن
            non_padded_items_in_sequence = [item for item in u_input_single_items_padded if item !=0]

            for i_idx in np.arange(len(non_padded_items_in_sequence) - 1):
                item_curr = non_padded_items_in_sequence[i_idx]
                item_next = non_padded_items_in_sequence[i_idx+1]
                
                # item_curr و item_next باید در item_to_idx_map باشند چون از non_padded_items_in_sequence آمده‌اند
                # و node_unique_for_graph از همین آیتم‌ها ساخته شده.
                u = item_to_idx_map[item_curr]
                v = item_to_idx_map[item_next]
                u_A[u][v] += 1
            
            # نرمال‌سازی ماتریس همجواری
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[u_sum_in == 0] = 1 # جلوگیری از تقسیم بر صفر
            u_A_in = np.divide(u_A, u_sum_in)
            
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[u_sum_out == 0] = 1 # جلوگیری از تقسیم بر صفر
            u_A_out_transposed = np.divide(u_A.transpose(), u_sum_out) # تقسیم بر u_sum_out که برای هر سطر (گره مبدا) است
            u_A_out = u_A_out_transposed.transpose() # برگرداندن به حالت اولیه

            A_list.append(np.concatenate([u_A_in, u_A_out], axis=1)) # الحاق ماتریس ورودی و خروجی نرمال شده

            # ساخت آلیاس ورودی‌ها: نگاشت هر آیتم در توالی پد شده اصلی به ایندکسش در گره‌های یکتا
            alias_for_current_seq = []
            for item_val_in_seq_padded in u_input_single_items_padded:
                if item_val_in_seq_padded == 0: # اگر پدینگ است
                    alias_for_current_seq.append(0) # آلیاس پدینگ هم 0 در نظر گرفته می‌شود (باید با padding_idx مدل همخوان باشد)
                elif item_val_in_seq_padded in item_to_idx_map:
                    alias_for_current_seq.append(item_to_idx_map[item_val_in_seq_padded])
                else: 
                    # این حالت نباید رخ دهد اگر item_to_idx_map شامل همه آیتم‌های غیر پدینگ باشد
                    alias_for_current_seq.append(0) # مدیریت خطا
            alias_inputs_list.append(alias_for_current_seq)
            
        # current_time_diffs_batch_padded از ورودی تابع می‌آید و از قبل پد شده و احتمالا augment شده
        # پس نیازی به پردازش بیشتر روی آن نیست، فقط تبدیل به آرایه NumPy
            
        return np.array(alias_inputs_list), np.array(A_list), np.array(items_list_unique_nodes), \
               np.array(mask_list_for_view), np.array(positions_list_for_view), \
               np.array(current_time_diffs_batch_padded, dtype=np.float32)


    def get_slice(self, batch_indices, ssl_item_drop_prob=0.2):
        if len(batch_indices) == 0:
            # برگرداندن آرایه‌های خالی با شکل و نوع داده مناسب
            empty_alias = np.array([], dtype=np.int64)
            empty_A = np.array([], dtype=np.float32)
            empty_items_unique = np.array([], dtype=np.int64)
            empty_mask_ssl = np.array([], dtype=np.int64)
            empty_positions = np.array([], dtype=np.int64)
            empty_targets = np.array([], dtype=np.int64)
            empty_mask_main = np.array([], dtype=np.int64)
            empty_time_diffs = np.array([], dtype=np.float32)

            data_tuple = (empty_alias, empty_A, empty_items_unique, empty_mask_ssl, empty_positions)
            return data_tuple, data_tuple, empty_targets, empty_mask_main, empty_time_diffs, empty_time_diffs

        if isinstance(batch_indices, np.ndarray):
            batch_indices = batch_indices.tolist()
        
        # گرفتن داده‌های پد شده برای بچ فعلی از self.inputs, self.targets, self.mask, self.positions, self.time_diffs_padded
        batch_inputs_padded = self.inputs[batch_indices]
        batch_targets_np = np.array(self.targets)[batch_indices] if isinstance(self.targets, list) else self.targets[batch_indices]
        batch_mask_main_np = self.mask[batch_indices] # ماسک اصلی برای آیتم‌های view1 (بدون دراپ زیاد)
        # batch_positions_np = self.positions[batch_indices] # موقعیت‌ها معمولا برای view اصلی استفاده می‌شوند
        batch_time_diffs_padded_np = self.time_diffs_padded[batch_indices]

        # ایجاد view 1 (معمولا با drop_prob کم یا صفر برای وظیفه اصلی)
        # و view 2 (با drop_prob بیشتر برای SSL)
        
        inputs_v1_augmented_padded_list = []
        time_diffs_v1_augmented_padded_list = []
        inputs_v2_augmented_padded_list = []
        time_diffs_v2_augmented_padded_list = []

        for i in range(len(batch_inputs_padded)):
            current_seq_items_padded = batch_inputs_padded[i]
            current_seq_time_diffs_padded = batch_time_diffs_padded_np[i]
            
            # View 1 (main task view - معمولا بدون دراپ یا با دراپ خیلی کم)
            # اینجا فرض می‌کنیم برای v1 دراپ اعمال نمی‌کنیم و از خود داده اصلی پد شده استفاده می‌کنیم.
            # یا می‌توانیم از _augment_sequence_item_dropout با drop_prob=0.0 استفاده کنیم.
            # برای سازگاری با TAGNN که از داده اصلی برای main loss استفاده می‌کند:
            inputs_v1_augmented_padded_list.append(list(current_seq_items_padded))
            time_diffs_v1_augmented_padded_list.append(list(current_seq_time_diffs_padded))

            # View 2 (SSL view)
            v2_aug_seq, v2_aug_time = self._augment_sequence_item_dropout(
                current_seq_items_padded, 
                current_seq_time_diffs_padded, 
                ssl_item_drop_prob
            )
            inputs_v2_augmented_padded_list.append(v2_aug_seq)
            time_diffs_v2_augmented_padded_list.append(v2_aug_time)


        # تبدیل داده‌های augment شده و پد شده به فرمت گراف برای هر view
        alias_v1, A_v1, items_v1_unique, mask_v1_ssl, positions_v1, time_diffs_v1_final = \
            self._get_graph_data_for_view(inputs_v1_augmented_padded_list, time_diffs_v1_augmented_padded_list)
        
        alias_v2, A_v2, items_v2_unique, mask_v2_ssl, positions_v2, time_diffs_v2_final = \
            self._get_graph_data_for_view(inputs_v2_augmented_padded_list, time_diffs_v2_augmented_padded_list)
        
        # batch_mask_main_np از self.mask گرفته شده که متناظر با inputs_v1_augmented_padded_list است (اگر v1 دراپ نداشته باشد).
        # اگر v1 هم augment شود، ماسک اصلی باید بر اساس آن view ساخته شود.
        # در اینجا فرض شده batch_mask_main_np برای view1 (بدون دراپ یا با دراپ کم) معتبر است.

        return (alias_v1, A_v1, items_v1_unique, mask_v1_ssl, positions_v1), \
               (alias_v2, A_v2, items_v2_unique, mask_v2_ssl, positions_v2), \
               np.array(batch_targets_np), np.array(batch_mask_main_np), \
               time_diffs_v1_final, time_diffs_v2_final