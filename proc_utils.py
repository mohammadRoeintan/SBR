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
    if not us_lens: # Handle case where all_usr_pois is empty
        len_max = max_len if max_len > 0 else 0

    us_pois = []
    for upois in all_usr_pois:
        current_len = len(upois)
        # Ensure padding length is not negative if current_len > len_max (due to dynamic len_max)
        padding_len = max(0, len_max - min(current_len, len_max))
        us_pois.append(upois[:min(current_len, len_max)] + item_tail * padding_len)

    us_msks = [[1] * min(le, len_max) + [0] * max(0, (len_max - min(le, len_max))) for le in us_lens]
    
    us_pos = [
        list(range(1, min(le, len_max) + 1)) + [0] * max(0, (len_max - min(le, len_max)))
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
            # Ensure each element in time_data is a list
            self.time_data_raw = [list(td) if not isinstance(td, list) else td for td in time_data]


        self.length = len(self.raw_inputs)
        self.shuffle = shuffle
        self.opt = opt
        
        # Determine len_max based on current data in this Dataset instance
        current_max_len_in_data = 0
        if self.raw_inputs:
            current_max_len_in_data = max(len(s) for s in self.raw_inputs) if self.raw_inputs else 0
        
        if opt and hasattr(opt, 'max_len') and opt.max_len > 0:
            # If opt.max_len is specified, it acts as an upper bound for padding for this dataset instance
            self.len_max_for_padding = min(opt.max_len, current_max_len_in_data) if current_max_len_in_data > 0 else opt.max_len
            if current_max_len_in_data == 0 and opt.max_len > 0 : # If raw_inputs is empty but opt.max_len is set
                 self.len_max_for_padding = opt.max_len
            elif current_max_len_in_data > 0 and opt.max_len > 0:
                 self.len_max_for_padding = min(opt.max_len, current_max_len_in_data)
            elif current_max_len_in_data > 0 and opt.max_len == 0: # opt.max_len means auto, so use actual max
                 self.len_max_for_padding = current_max_len_in_data
            else: # both are 0
                 self.len_max_for_padding = 0

        else: # opt.max_len is not set or is 0, use the actual max length from data
            self.len_max_for_padding = current_max_len_in_data
        
        # Pad sequences to self.len_max_for_padding for this specific dataset (train, valid, or test)
        padded_inputs, padded_mask, padded_pos, _ = data_masks(self.raw_inputs, [0], self.len_max_for_padding)
        self.inputs = np.asarray(padded_inputs)
        self.mask = np.asarray(padded_mask)
        self.positions = np.asarray(padded_pos)
        
        self.time_diffs_padded = self._pad_and_prepare_time_diffs() # Uses self.len_max_for_padding

    def _pad_and_prepare_time_diffs(self):
        # This function pads time_diffs to self.len_max_for_padding
        padded_time_diffs_matrix = []
        for i, session_raw_input in enumerate(self.raw_inputs):
            current_session_time_diffs_orig = []
            if i < len(self.time_data_raw):
                current_session_time_diffs_orig = list(self.time_data_raw[i]) # Make a copy
            else: # Should not happen if time_data_raw is correctly initialized
                current_session_time_diffs_orig = [0.0] * len(session_raw_input)

            # Ensure time diffs match the length of raw input session *before* truncation/padding
            # This part aligns time_diffs with the actual items in the raw session
            effective_raw_len = len(session_raw_input)
            if len(current_session_time_diffs_orig) < effective_raw_len:
                current_session_time_diffs_orig.extend([0.0] * (effective_raw_len - len(current_session_time_diffs_orig)))
            elif len(current_session_time_diffs_orig) > effective_raw_len:
                current_session_time_diffs_orig = current_session_time_diffs_orig[:effective_raw_len]
            
            # Now, truncate or pad to self.len_max_for_padding
            final_padded_time_diffs_for_session = current_session_time_diffs_orig[:self.len_max_for_padding]
            if len(final_padded_time_diffs_for_session) < self.len_max_for_padding:
                final_padded_time_diffs_for_session.extend([0.0] * (self.len_max_for_padding - len(final_padded_time_diffs_for_session)))
            
            padded_time_diffs_matrix.append(final_padded_time_diffs_for_session)
            
        return np.array(padded_time_diffs_matrix, dtype=np.float32)
        
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            
            # It's crucial to shuffle all corresponding data structures
            self.raw_inputs = [self.raw_inputs[i] for i in shuffled_arg]
            self.targets = [self.targets[i] for i in shuffled_arg]
            if hasattr(self, 'time_data_raw') and self.time_data_raw:
                 self.time_data_raw = [self.time_data_raw[i] for i in shuffled_arg]

            # Re-pad after shuffling, using the dataset's len_max_for_padding
            current_padded_inputs, current_padded_mask, current_padded_pos, _ = data_masks(self.raw_inputs, [0], self.len_max_for_padding)
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
            if start_idx < end_idx : # Ensure slice is not empty
                 slices.append(np.arange(start_idx, end_idx))
        
        return slices

    def _augment_sequence_item_dropout(self, seq_items_padded_input, seq_time_diffs_padded_input, drop_prob):
        # seq_items_padded_input and seq_time_diffs_padded_input are single sequences (1D arrays/lists)
        # Their length is determined by the batch's max length (which could be <= self.len_max_for_padding)
        
        current_sequence_length = len(seq_items_padded_input)
        
        temp_augmented_seq = []
        temp_augmented_time = []
        
        items_kept_count = 0
        original_non_pad_item_count = 0

        for i in range(current_sequence_length): # Iterate up to the length of the current sequence
            item = seq_items_padded_input[i]
            time_d = seq_time_diffs_padded_input[i]

            if item != 0: # It's an actual item, not padding
                original_non_pad_item_count +=1
                if random.random() > drop_prob: # Keep this item
                    temp_augmented_seq.append(item)
                    temp_augmented_time.append(time_d)
                    items_kept_count += 1
            # else: it's a padding item (0), it won't be added to temp_augmented_seq anyway
        
        # If all items were dropped (and there were items to begin with), keep at least one
        if items_kept_count == 0 and original_non_pad_item_count > 0:
            # Find the first non-padding item in the original input sequence and add it
            first_non_pad_idx = -1
            for idx, item_val in enumerate(seq_items_padded_input):
                if item_val != 0:
                    first_non_pad_idx = idx
                    break
            if first_non_pad_idx != -1: # Should always be true if original_non_pad_item_count > 0
                temp_augmented_seq = [seq_items_padded_input[first_non_pad_idx]]
                temp_augmented_time = [seq_time_diffs_padded_input[first_non_pad_idx]]

        # Pad the augmented sequence back to the original input sequence's length (current_sequence_length)
        # This ensures that sequences within the same batch retain the same augmented length before _get_graph_data_for_view
        padded_augmented_seq = temp_augmented_seq + [0] * (current_sequence_length - len(temp_augmented_seq))
        padded_augmented_time = temp_augmented_time + [0.0] * (current_sequence_length - len(temp_augmented_time))
        
        return padded_augmented_seq, padded_augmented_time


    def _get_graph_data_for_view(self, current_inputs_batch_padded_items, current_time_diffs_batch_padded):
        # current_inputs_batch_padded_items is a batch of sequences, already padded to the max_len of THIS BATCH
        
        batch_size = len(current_inputs_batch_padded_items)
        # The max_len for this specific batch (could be different from self.len_max_for_padding)
        batch_current_max_seq_len = 0
        if batch_size > 0:
            batch_current_max_seq_len = len(current_inputs_batch_padded_items[0])


        items_list_unique_nodes, A_list, alias_inputs_list = [], [], []
        mask_list_for_view = [] 
        positions_list_for_view = []
        
        batch_max_n_node_in_this_batch = 0 # Max unique nodes in any sequence IN THIS BATCH
        temp_n_nodes_for_batch = []
        for u_input_single_items_padded in current_inputs_batch_padded_items:
            unique_nodes_in_seq = np.unique(np.array(u_input_single_items_padded))
            unique_nodes_in_seq = unique_nodes_in_seq[unique_nodes_in_seq != 0] # Exclude padding
            num_unique = len(unique_nodes_in_seq)
            temp_n_nodes_for_batch.append(num_unique if num_unique > 0 else 1) # Ensure at least 1 for empty sequences after augmentation
        
        if temp_n_nodes_for_batch: # If the batch is not empty
            batch_max_n_node_in_this_batch = np.max(temp_n_nodes_for_batch)
        else: # Batch is empty
            batch_max_n_node_in_this_batch = 1 # Default to 1 if batch is empty

        for idx, u_input_single_items_padded in enumerate(current_inputs_batch_padded_items):
            # u_input_single_items_padded is a single sequence from the batch
            # Its length is batch_current_max_seq_len
            
            current_mask_for_sequence = [1 if item != 0 else 0 for item in u_input_single_items_padded]
            mask_list_for_view.append(current_mask_for_sequence)
            
            current_pos_seq_for_sequence = []
            effective_len = 0
            for item_val in u_input_single_items_padded: # Iterate through the current sequence
                if item_val != 0:
                    effective_len += 1
                    current_pos_seq_for_sequence.append(effective_len)
                else:
                    current_pos_seq_for_sequence.append(0)
            positions_list_for_view.append(current_pos_seq_for_sequence)

            # Unique nodes for graph construction for THIS sequence
            node_unique_for_graph = np.unique(np.array(u_input_single_items_padded))
            node_unique_for_graph = node_unique_for_graph[node_unique_for_graph != 0] # Exclude padding
            
            if len(node_unique_for_graph) == 0: # If sequence became empty after augmentation
                 # Pad unique nodes to batch_max_n_node_in_this_batch
                 items_list_unique_nodes.append([0] * batch_max_n_node_in_this_batch)
                 # Adjacency matrix will be all zeros
                 A_list.append(np.zeros((batch_max_n_node_in_this_batch, batch_max_n_node_in_this_batch * 2)))
                 # Alias inputs will be all zeros, padded to batch_current_max_seq_len
                 alias_inputs_list.append([0] * batch_current_max_seq_len)
                 continue

            # Pad unique nodes for this sequence to batch_max_n_node_in_this_batch
            padded_unique_nodes_for_sequence = node_unique_for_graph.tolist() + [0] * (batch_max_n_node_in_this_batch - len(node_unique_for_graph))
            items_list_unique_nodes.append(padded_unique_nodes_for_sequence)
            
            # Adjacency matrix, size based on batch_max_n_node_in_this_batch
            u_A = np.zeros((batch_max_n_node_in_this_batch, batch_max_n_node_in_this_batch))
            # Map item_id to its index within node_unique_for_graph (local to this sequence's unique items)
            item_to_idx_map = {item_id: k for k, item_id in enumerate(node_unique_for_graph)}

            # Get non-padded items from the current sequence to build the graph
            non_padded_items_in_current_sequence = [item for item in u_input_single_items_padded if item !=0]

            for i_adj_idx in np.arange(len(non_padded_items_in_current_sequence) - 1):
                item_curr = non_padded_items_in_current_sequence[i_adj_idx]
                item_next = non_padded_items_in_current_sequence[i_adj_idx+1]
                
                # Map these items to their indices in the unique node list for this sequence
                u = item_to_idx_map[item_curr]
                v = item_to_idx_map[item_next]
                u_A[u][v] += 1 # Build adjacency matrix based on unique node indices
            
            # Normalize adjacency matrix
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[u_sum_in == 0] = 1 # Avoid division by zero
            u_A_in = np.divide(u_A, u_sum_in)
            
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[u_sum_out == 0] = 1 # Avoid division by zero
            u_A_out_transposed = np.divide(u_A.transpose(), u_sum_out) # Divide columns by u_sum_out
            u_A_out = u_A_out_transposed.transpose() # Transpose back

            A_list.append(np.concatenate([u_A_in, u_A_out], axis=1))

            # Alias inputs: map items in the original padded sequence to their *local* unique node index
            alias_for_current_sequence = []
            for item_val_in_seq_padded in u_input_single_items_padded: # Iterate through the current sequence
                if item_val_in_seq_padded == 0: # Padding item
                    alias_for_current_sequence.append(0) # Map padding to 0 (or some other convention if 0 is a valid unique node index, though typically not)
                elif item_val_in_seq_padded in item_to_idx_map: # Item is in unique map for this sequence
                    alias_for_current_sequence.append(item_to_idx_map[item_val_in_seq_padded])
                else: # Item not in unique map (e.g., if it was a padding item in original data that got here)
                      # This case should ideally not happen if node_unique_for_graph is built from u_input_single_items_padded
                    alias_for_current_sequence.append(0) 
            alias_inputs_list.append(alias_for_current_sequence)
            
        return np.array(alias_inputs_list), np.array(A_list), np.array(items_list_unique_nodes), \
               np.array(mask_list_for_view), np.array(positions_list_for_view), \
               np.array(current_time_diffs_batch_padded, dtype=np.float32)


    def get_slice(self, batch_indices, ssl_item_drop_prob=0.2):
        if len(batch_indices) == 0:
            # Return empty arrays with correct dtypes and number of dimensions if batch is empty
            # The shapes here are (0, X) or (0, X, Y) where X, Y are typical feature dimensions
            # This needs to match what the model expects for an empty batch if it were to proceed
            # However, typically, empty batches are skipped earlier.
            # For robustness, define shapes based on expected dimensions if known (e.g., self.len_max_for_padding)
            
            # Max length for this slice, could be 0 if batch_indices is empty
            # This will be used to shape the empty arrays
            # If we know the max length of sequences in this batch, use it.
            # If batch_indices is empty, there's no max length from data.
            # Default to a small length or 0, but ensure subsequent code handles empty tensors.
            # In practice, the calling code (train_test) should skip if len(batch_indices)==0
            
            # Fallback: if the dataset has a defined padding length, use that.
            # Otherwise, if it is truly an empty slice from an empty dataset, len_for_empty_slice = 0
            len_for_empty_slice = self.len_max_for_padding if hasattr(self, 'len_max_for_padding') else 0
            
            # Fallback for max_n_node, needed for A's shape
            # This is tricky if the batch is truly empty. Assume 1 as a minimal dimension.
            max_n_node_for_empty_slice = 1 


            empty_alias = np.array([], dtype=np.int64).reshape(0, len_for_empty_slice)
            empty_A = np.array([], dtype=np.float32).reshape(0, max_n_node_for_empty_slice, max_n_node_for_empty_slice * 2)
            empty_items_unique = np.array([], dtype=np.int64).reshape(0, max_n_node_for_empty_slice)
            empty_mask_ssl = np.array([], dtype=np.int64).reshape(0, len_for_empty_slice)
            empty_positions = np.array([], dtype=np.int64).reshape(0, len_for_empty_slice)
            
            empty_targets = np.array([], dtype=np.int64)
            empty_mask_main = np.array([], dtype=np.int64).reshape(0, len_for_empty_slice)
            empty_time_diffs = np.array([], dtype=np.float32).reshape(0, len_for_empty_slice)

            data_tuple = (empty_alias, empty_A, empty_items_unique, empty_mask_ssl, empty_positions)
            return data_tuple, data_tuple, empty_targets, empty_mask_main, empty_time_diffs, empty_time_diffs

        # Convert batch_indices to list if it's a numpy array
        if isinstance(batch_indices, np.ndarray):
            batch_indices = batch_indices.tolist()
        
        # These are already padded to self.len_max_for_padding (max length of the *entire dataset* split)
        batch_inputs_from_dataset = self.inputs[batch_indices]
        batch_targets_np = np.array(self.targets)[batch_indices] if isinstance(self.targets, list) else self.targets[batch_indices]
        batch_mask_main_from_dataset = self.mask[batch_indices] # Mask corresponding to original, non-augmented data
        batch_time_diffs_from_dataset = self.time_diffs_padded[batch_indices]


        # For augmentation and graph construction, we might operate on sequences
        # potentially shorter than self.len_max_for_padding if this batch's true max length is smaller.
        # However, _augment_sequence_item_dropout and _get_graph_data_for_view
        # should correctly handle the length of sequences passed to them.
        # The sequences passed to _augment will be rows from batch_inputs_from_dataset.

        inputs_v1_augmented_padded_list = []
        time_diffs_v1_augmented_padded_list = []
        inputs_v2_augmented_padded_list = []
        time_diffs_v2_augmented_padded_list = []

        for i in range(len(batch_inputs_from_dataset)):
            # current_seq_items_padded and current_seq_time_diffs_padded are 1D arrays (single sequences)
            current_seq_items_padded = batch_inputs_from_dataset[i]
            current_seq_time_diffs_padded = batch_time_diffs_from_dataset[i]
            
            # v1 is typically the original sequence for main loss (no item dropout)
            # The _augment_sequence_item_dropout with 0 prob should return the original sequence,
            # but correctly padded to its own length.
            v1_aug_seq, v1_aug_time = self._augment_sequence_item_dropout(
                current_seq_items_padded, 
                current_seq_time_diffs_padded, 
                0.0 # No item dropout for v1
            )
            inputs_v1_augmented_padded_list.append(v1_aug_seq)
            time_diffs_v1_augmented_padded_list.append(v1_aug_time)

            # v2 is the augmented view for SSL
            v2_aug_seq, v2_aug_time = self._augment_sequence_item_dropout(
                current_seq_items_padded, 
                current_seq_time_diffs_padded, 
                ssl_item_drop_prob # Apply SSL item dropout for v2
            )
            inputs_v2_augmented_padded_list.append(v2_aug_seq)
            time_diffs_v2_augmented_padded_list.append(v2_aug_time)

        # _get_graph_data_for_view will internally determine the max_n_node and max_seq_len for THIS BATCH
        # of (potentially augmented) sequences.
        alias_v1, A_v1, items_v1_unique, mask_v1_ssl, positions_v1, time_diffs_v1_final = \
            self._get_graph_data_for_view(inputs_v1_augmented_padded_list, time_diffs_v1_augmented_padded_list)
        
        alias_v2, A_v2, items_v2_unique, mask_v2_ssl, positions_v2, time_diffs_v2_final = \
            self._get_graph_data_for_view(inputs_v2_augmented_padded_list, time_diffs_v2_augmented_padded_list)
        
        # batch_mask_main_from_dataset is the mask for the original sequences from the dataset.
        # This should be used for the main prediction loss. Its length corresponds to self.inputs padding.
        # If alias_v1 has a different length due to _get_graph_data_for_view using batch-specific max_len,
        # ensure the model's attention mask logic handles this.
        # The `mask_main_np` returned should align with the sequences used for main prediction (v1).
        # `mask_v1_ssl` is the mask for the (potentially length-adjusted) v1 sequences.
        # For main loss, we need a mask that matches the length of sequences in `alias_v1` or `final_seq_hidden_v1`.
        # `mask_v1_ssl` generated by _get_graph_data_for_view is the correct mask for `final_seq_hidden_v1`.

        return (alias_v1, A_v1, items_v1_unique, mask_v1_ssl, positions_v1), \
               (alias_v2, A_v2, items_v2_unique, mask_v2_ssl, positions_v2), \
               np.array(batch_targets_np), np.array(mask_v1_ssl), \
               time_diffs_v1_final, time_diffs_v2_final
