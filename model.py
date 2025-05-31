import numpy as np
import torch
import random
import bisect
import numpy as np
import torch
import random
from collections import defaultdict

def split_validation(data, valid_portion=0.1):
    time_data = data[0]
    train_set_x = data[1]
    train_set_y = data[2]
    
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    
    n_train = int(np.round(n_samples * (1. - valid_portion))
    
    valid_set_x = [train_set_x[i] for i in sidx[n_train:] if i < len(train_set_x)]
    valid_set_y = [train_set_y[i] for i in sidx[n_train:] if i < len(train_set_y)]
    
    train_set_x = [train_set_x[i] for i in sidx[:n_train] if i < len(train_set_x)]
    train_set_y = [train_set_y[i] for i in sidx[:n_train] if i < len(train_set_y)]
    
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def data_masks(all_usr_pois, item_tail, max_len):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = min(max(us_lens), max_len) if max_len > 0 else max(us_lens)
    us_pois = [upois[:len_max] + item_tail * (len_max - min(len(upois), len_max)) for upois in all_usr_pois]
    us_msks = [[1] * min(le, len_max) + [0] * (len_max - min(le, len_max)) for le in us_lens]
    us_pos = [list(range(1, min(le, len_max)+1)) + [0] * (len_max - min(le, len_max)) for le in us_lens]
    return us_pois, us_msks, us_pos, len_max

class Dataset():
    def __init__(self, data, time_data=None, shuffle=False, opt=None):
        self.raw_inputs = [list(seq) for seq in data[0]]
        
        if isinstance(data[1], np.ndarray):
            self.targets = data[1].tolist()
        else:
            self.targets = data[1]
            
        self.time_data = time_data if time_data else [np.zeros(len(seq)) for seq in self.raw_inputs]
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
        self.time_diffs = self._compute_time_diffs()

    def _compute_time_diffs(self):
        time_diffs = []
        for i, session in enumerate(self.raw_inputs):
            td_session = []
            time_stamps = self.time_data[i]
            for j in range(1, len(session)):
                if j < len(time_stamps):
                    td_session.append(time_stamps[j] - time_stamps[j-1])
                else:
                    td_session.append(0)
            
            if len(td_session) < self.len_max:
                td_session += [0] * (self.len_max - len(td_session))
            else:
                td_session = td_session[:self.len_max]
            time_diffs.append(td_session)
        return np.array(time_diffs, dtype=np.float32)

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw_inputs = [self.raw_inputs[i] for i in shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self.time_diffs = self.time_diffs[shuffled_arg]
            
            current_padded_inputs, current_padded_mask, current_padded_pos, _ = data_masks(self.raw_inputs, [0], self.len_max)
            self.inputs = np.asarray(current_padded_inputs)
            self.mask = np.asarray(current_padded_mask)
            self.positions = np.asarray(current_padded_pos)

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        if self.length == 0:
             return []
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        slices = [s for s in slices if len(s) > 0]
        return slices

    def _augment_sequence_item_dropout(self, seq, time_diff, drop_prob, max_len):
        if drop_prob == 0:
            return seq[:max_len] + [0] * (max_len - len(seq)), time_diff[:max_len] + [0] * (max_len - len(time_diff))
        
        augmented_seq = []
        augmented_time = []
        for i, item in enumerate(seq):
            if random.random() > drop_prob:
                augmented_seq.append(item)
                if i < len(time_diff):
                    augmented_time.append(time_diff[i])
            if len(augmented_seq) >= max_len:
                break
                
        if len(augmented_seq) == 0:
            augmented_seq = [seq[0]]
            augmented_time = [time_diff[0] if time_diff else 0]
            
        augmented_seq = augmented_seq[:max_len]
        augmented_time = augmented_time[:max_len]
        
        return augmented_seq + [0] * (max_len - len(augmented_seq)), augmented_time + [0] * (max_len - len(augmented_time))

    def _get_graph_data_for_view(self, current_inputs_batch_padded_items, time_diffs_batch):
        items_list, A_list, alias_inputs_list = [], [], []
        mask_list_for_view = [] 
        positions_list_for_view = []
        time_diffs_list = []

        batch_max_n_node = 0
        temp_n_nodes_for_batch = []
        for u_input_single_items in current_inputs_batch_padded_items:
            unique_nodes_in_seq = np.unique(np.array(u_input_single_items))
            unique_nodes_in_seq = unique_nodes_in_seq[unique_nodes_in_seq != 0]
            num_unique = len(unique_nodes_in_seq)
            temp_n_nodes_for_batch.append(num_unique if num_unique > 0 else 1)
        
        batch_max_n_node = np.max(temp_n_nodes_for_batch) if temp_n_nodes_for_batch else 1

        for idx, u_input_single_items in enumerate(current_inputs_batch_padded_items):
            current_mask = [1 if item != 0 else 0 for item in u_input_single_items]
            mask_list_for_view.append(current_mask)
            
            current_pos_seq = []
            effective_len = 0
            for item_val in u_input_single_items:
                if item_val != 0:
                    effective_len += 1
                    current_pos_seq.append(effective_len)
                else:
                    current_pos_seq.append(0)
            positions_list_for_view.append(current_pos_seq)

            node = np.unique(np.array(u_input_single_items))
            node = node[node != 0]
            if len(node) == 0: 
                node = np.array([0])
            
            items_list.append(node.tolist() + (batch_max_n_node - len(node)) * [0])
            
            u_A = np.zeros((batch_max_n_node, batch_max_n_node))
            item_to_idx_map = {item_id: k for k, item_id in enumerate(node)}

            for i_idx in np.arange(len(u_input_single_items) - 1):
                item_curr = u_input_single_items[i_idx]
                item_next = u_input_single_items[i_idx+1]

                if item_curr == 0 or item_next == 0: 
                    continue
                
                if item_curr in item_to_idx_map and item_next in item_to_idx_map:
                    u = item_to_idx_map[item_curr]
                    v = item_to_idx_map[item_next]
                    u_A[u][v] += 1
            
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[u_sum_in == 0] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[u_sum_out == 0] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A_out = u_A_out.transpose()
            A_list.append(np.concatenate([u_A_in, u_A_out], axis=1))

            alias_for_current_seq = []
            for item_val_in_seq in u_input_single_items:
                if item_val_in_seq == 0: 
                    alias_for_current_seq.append(0)
                elif item_val_in_seq in item_to_idx_map:
                    alias_for_current_seq.append(item_to_idx_map[item_val_in_seq])
                else: 
                    alias_for_current_seq.append(0)
            alias_inputs_list.append(alias_for_current_seq)
            
            time_diffs_list.append(time_diffs_batch[idx])
            
        return np.array(alias_inputs_list), np.array(A_list), np.array(items_list), \
               np.array(mask_list_for_view), np.array(positions_list_for_view), np.array(time_diffs_list)

    def get_slice(self, batch_indices, ssl_item_drop_prob=0.2):
        if len(batch_indices) == 0:
            return (np.array([], dtype=np.int64),  np.array([], dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64),np.array([], dtype=np.int64)), (np.array([], dtype=np.int64), np.array([], dtype=np.float32),np.array([], dtype=np.int64),np.array([], dtype=np.int64), np.array([], dtype=np.int64)), np.array([], dtype=np.int64), np.array([], dtype=np.int64),np.array([], dtype=np.float32),np.array([], dtype=np.float32)

        batch_raw_inputs_unpadded = [self.raw_inputs[idx] for idx in batch_indices]
        batch_targets = self.targets[batch_indices]
        batch_time_diffs = [self.time_diffs[idx] for idx in batch_indices]
        
        batch_us_lens = [len(s) for s in batch_raw_inputs_unpadded]
        current_batch_max_len = min(max(batch_us_lens) if batch_us_lens else 0, self.len_max)
        if current_batch_max_len == 0 and len(batch_raw_inputs_unpadded) > 0:
             current_batch_max_len = 1

        inputs_v1_padded_items = []
        time_diffs_v1 = []
        for i, seq in enumerate(batch_raw_inputs_unpadded):
            padded_seq, padded_time = self._augment_sequence_item_dropout(seq, batch_time_diffs[i], 0.0, current_batch_max_len)
            inputs_v1_padded_items.append(padded_seq)
            time_diffs_v1.append(padded_time)
            
        inputs_v2_padded_items = []
        time_diffs_v2 = []
        for i, seq in enumerate(batch_raw_inputs_unpadded):
            padded_seq, padded_time = self._augment_sequence_item_dropout(seq, batch_time_diffs[i], ssl_item_drop_prob, current_batch_max_len)
            inputs_v2_padded_items.append(padded_seq)
            time_diffs_v2.append(padded_time)

        alias_v1, A_v1, items_v1, mask_v1_ssl, positions_v1, time_diffs_v1 = self._get_graph_data_for_view(
            inputs_v1_padded_items, time_diffs_v1
        )
        alias_v2, A_v2, items_v2, mask_v2_ssl, positions_v2, time_diffs_v2 = self._get_graph_data_for_view(
            inputs_v2_padded_items, time_diffs_v2
        )

        _, batch_mask_main_list, _, _ = data_masks(batch_raw_inputs_unpadded, [0], current_batch_max_len)
        
        return (alias_v1, A_v1, items_v1, mask_v1_ssl, positions_v1), \
               (alias_v2, A_v2, items_v2, mask_v2_ssl, positions_v2), \
               batch_targets, np.array(batch_mask_main_list), \
               np.array(time_diffs_v1), np.array(time_diffs_v2)
