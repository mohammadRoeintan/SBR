import numpy as np
import random
from collections import defaultdict

def split_validation(data, valid_portion=0.1):
    train_set_x = data[0]
    train_set_y = data[1]
    
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    
    train_x = [train_set_x[i] for i in sidx[:n_train]]
    train_y = [train_set_y[i] for i in sidx[:n_train]]
    
    valid_x = [train_set_x[i] for i in sidx[n_train:]]
    valid_y = [train_set_y[i] for i in sidx[n_train:]]
    
    return (train_x, train_y), (valid_x, valid_y)

def data_masks(all_usr_pois, item_tail, max_len):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = min(max(us_lens), max_len) if max_len > 0 else max(us_lens)
    if not us_lens:
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
        self.raw_inputs = [list(seq) for seq in data[0]]
        
        if isinstance(data[1], np.ndarray):
            self.targets = data[1].tolist()
        else:
            self.targets = data[1]
            
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
        
        self.time_diffs_padded = self._pad_and_prepare_time_diffs()

    def _pad_and_prepare_time_diffs(self):
        padded_time_diffs_matrix = []
        for i, session_raw_input in enumerate(self.raw_inputs):
            if i < len(self.time_data_raw):
                current_session_time_diffs = list(self.time_data_raw[i])
            else:
                current_session_time_diffs = [0.0] * len(session_raw_input)

            if len(current_session_time_diffs) != len(session_raw_input) and len(session_raw_input) > 0 :
                if len(current_session_time_diffs) < len(session_raw_input):
                    current_session_time_diffs.extend([0.0] * (len(session_raw_input) - len(current_session_time_diffs)))
                else:
                    current_session_time_diffs = current_session_time_diffs[:len(session_raw_input)]
            
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
            self.time_data_raw = [self.time_data_raw[i] for i in shuffled_indices]

            current_padded_inputs, current_padded_mask, current_padded_pos, _ = data_masks(self.raw_inputs, [0], self.len_max)
            self.inputs = np.asarray(current_padded_inputs)
            self.mask = np.asarray(current_padded_mask)
            self.positions = np.asarray(current_padded_pos)
            self.time_diffs_padded = self._pad_and_prepare_time_diffs()

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        
        slices = []
        for i in range(n_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.length)
            if start_idx < end_idx :
                 slices.append(np.arange(start_idx, end_idx))
        
        return slices

    def _augment_sequence_item_dropout(self, seq_items_padded, seq_time_diffs_padded, drop_prob):
        augmented_seq_items = []
        augmented_time_diffs = []
        
        non_pad_items_indices = [i for i, item in enumerate(seq_items_padded) if item != 0]

        if not non_pad_items_indices:
            return list(seq_items_padded), list(seq_time_diffs_padded)

        for i in range(len(seq_items_padded)):
            item = seq_items_padded[i]
            time_d = seq_time_diffs_padded[i]
            
            if item == 0:
                augmented_seq_items.append(item)
                augmented_time_diffs.append(time_d)
                continue

            if random.random() > drop_prob:
                augmented_seq_items.append(item)
                augmented_time_diffs.append(time_d)
        
        final_augmented_seq_items = []
        final_augmented_time_diffs = []
        
        temp_augmented_seq = []
        temp_augmented_time = []
        
        original_length_before_pad = 0
        for item_val in seq_items_padded:
            if item_val != 0:
                original_length_before_pad +=1
        
        if original_length_before_pad == 0:
            return list(seq_items_padded), list(seq_time_diffs_padded)

        items_kept_count = 0
        for i in range(self.len_max):
            item = seq_items_padded[i]
            time_d = seq_time_diffs_padded[i]

            if item != 0:
                if random.random() > drop_prob:
                    temp_augmented_seq.append(item)
                    temp_augmented_time.append(time_d)
                    items_kept_count+=1
            
        if items_kept_count == 0 and original_length_before_pad > 0:
             first_non_pad_idx = -1
             for idx, item_val in enumerate(seq_items_padded):
                 if item_val != 0:
                     first_non_pad_idx = idx
                     break
             if first_non_pad_idx != -1:
                 temp_augmented_seq = [seq_items_padded[first_non_pad_idx]]
                 temp_augmented_time = [seq_time_diffs_padded[first_non_pad_idx]]

        padded_augmented_seq = temp_augmented_seq + [0] * (self.len_max - len(temp_augmented_seq))
        padded_augmented_time = temp_augmented_time + [0.0] * (self.len_max - len(temp_augmented_time))
        
        return padded_augmented_seq, padded_augmented_time


    def _get_graph_data_for_view(self, current_inputs_batch_padded_items, current_time_diffs_batch_padded):
        batch_size = len(current_inputs_batch_padded_items)
        items_list_unique_nodes, A_list, alias_inputs_list = [], [], []
        mask_list_for_view = [] 
        positions_list_for_view = []
        
        batch_max_n_node = 0
        temp_n_nodes_for_batch = []
        for u_input_single_items_padded in current_inputs_batch_padded_items:
            unique_nodes_in_seq = np.unique(np.array(u_input_single_items_padded))
            unique_nodes_in_seq = unique_nodes_in_seq[unique_nodes_in_seq != 0]
            num_unique = len(unique_nodes_in_seq)
            temp_n_nodes_for_batch.append(num_unique if num_unique > 0 else 1)
        
        if temp_n_nodes_for_batch:
            batch_max_n_node = np.max(temp_n_nodes_for_batch)
        else:
            batch_max_n_node = 1

        for idx, u_input_single_items_padded in enumerate(current_inputs_batch_padded_items):
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

            node_unique_for_graph = np.unique(np.array(u_input_single_items_padded))
            node_unique_for_graph = node_unique_for_graph[node_unique_for_graph != 0]
            
            if len(node_unique_for_graph) == 0:
                 items_list_unique_nodes.append([0] * batch_max_n_node)
                 A_list.append(np.zeros((batch_max_n_node, batch_max_n_node * 2)))
                 alias_inputs_list.append([0] * self.len_max)
                 continue

            padded_unique_nodes = node_unique_for_graph.tolist() + [0] * (batch_max_n_node - len(node_unique_for_graph))
            items_list_unique_nodes.append(padded_unique_nodes)
            
            u_A = np.zeros((batch_max_n_node, batch_max_n_node))
            item_to_idx_map = {item_id: k for k, item_id in enumerate(node_unique_for_graph)}

            non_padded_items_in_sequence = [item for item in u_input_single_items_padded if item !=0]

            for i_idx in np.arange(len(non_padded_items_in_sequence) - 1):
                item_curr = non_padded_items_in_sequence[i_idx]
                item_next = non_padded_items_in_sequence[i_idx+1]
                
                u = item_to_idx_map[item_curr]
                v = item_to_idx_map[item_next]
                u_A[u][v] += 1
            
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[u_sum_in == 0] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[u_sum_out == 0] = 1
            u_A_out_transposed = np.divide(u_A.transpose(), u_sum_out)
            u_A_out = u_A_out_transposed.transpose()

            A_list.append(np.concatenate([u_A_in, u_A_out], axis=1))

            alias_for_current_seq = []
            for item_val_in_seq_padded in u_input_single_items_padded:
                if item_val_in_seq_padded == 0:
                    alias_for_current_seq.append(0)
                elif item_val_in_seq_padded in item_to_idx_map:
                    alias_for_current_seq.append(item_to_idx_map[item_val_in_seq_padded])
                else: 
                    alias_for_current_seq.append(0)
            alias_inputs_list.append(alias_for_current_seq)
            
        return np.array(alias_inputs_list), np.array(A_list), np.array(items_list_unique_nodes), \
               np.array(mask_list_for_view), np.array(positions_list_for_view), \
               np.array(current_time_diffs_batch_padded, dtype=np.float32)


    def get_slice(self, batch_indices, ssl_item_drop_prob=0.2):
        if len(batch_indices) == 0:
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
        
        batch_inputs_padded = self.inputs[batch_indices]
        batch_targets_np = np.array(self.targets)[batch_indices] if isinstance(self.targets, list) else self.targets[batch_indices]
        batch_mask_main_np = self.mask[batch_indices]
        batch_time_diffs_padded_np = self.time_diffs_padded[batch_indices]

        inputs_v1_augmented_padded_list = []
        time_diffs_v1_augmented_padded_list = []
        inputs_v2_augmented_padded_list = []
        time_diffs_v2_augmented_padded_list = []

        for i in range(len(batch_inputs_padded)):
            current_seq_items_padded = batch_inputs_padded[i]
            current_seq_time_diffs_padded = batch_time_diffs_padded_np[i]
            
            inputs_v1_augmented_padded_list.append(list(current_seq_items_padded))
            time_diffs_v1_augmented_padded_list.append(list(current_seq_time_diffs_padded))

            v2_aug_seq, v2_aug_time = self._augment_sequence_item_dropout(
                current_seq_items_padded, 
                current_seq_time_diffs_padded, 
                ssl_item_drop_prob
            )
            inputs_v2_augmented_padded_list.append(v2_aug_seq)
            time_diffs_v2_augmented_padded_list.append(v2_aug_time)

        alias_v1, A_v1, items_v1_unique, mask_v1_ssl, positions_v1, time_diffs_v1_final = \
            self._get_graph_data_for_view(inputs_v1_augmented_padded_list, time_diffs_v1_augmented_padded_list)
        
        alias_v2, A_v2, items_v2_unique, mask_v2_ssl, positions_v2, time_diffs_v2_final = \
            self._get_graph_data_for_view(inputs_v2_augmented_padded_list, time_diffs_v2_augmented_padded_list)
        
        return (alias_v1, A_v1, items_v1_unique, mask_v1_ssl, positions_v1), \
               (alias_v2, A_v2, items_v2_unique, mask_v2_ssl, positions_v2), \
               np.array(batch_targets_np), np.array(batch_mask_main_np), \
               time_diffs_v1_final, time_diffs_v2_final