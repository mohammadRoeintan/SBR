##################################################################
# This code was adapted from https://github.com/CRIPAC-DIG/TAGNN #
##################################################################

import numpy as np
import torch # Added for torch.rand in augmentation

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens) if us_lens else 0 # Handle empty us_lens
    us_pois = [upois + item_tail * (len_max - le)
               for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Dataset():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        # Original inputs are kept as lists of lists for augmentation flexibility
        self.raw_inputs = [list(seq) for seq in inputs] # Store raw sequences
        self.targets = np.asarray(data[1])
        self.length = len(self.raw_inputs)
        self.shuffle = shuffle
        self.graph = graph # This seems unused, kept for compatibility

        # Determine len_max based on raw_inputs for consistent padding length
        us_lens = [len(upois) for upois in self.raw_inputs]
        self.len_max = max(us_lens) if us_lens else 0

        # Initial padding for self.inputs and self.mask (will be based on view 1 in get_slice)
        # This part is a bit tricky because augmentation changes content.
        # We'll use original sequences for the initial self.inputs and self.mask
        # and handle augmented sequence padding within get_slice or by ensuring
        # augmentations + padding happen correctly there.
        
        # For consistency, let's ensure self.inputs and self.mask reflect original data before augmentation.
        # data_masks can be used here directly if we assume augmentation happens on demand in get_slice.
        padded_inputs, padded_mask, _ = data_masks(self.raw_inputs, [0])
        self.inputs = np.asarray(padded_inputs) # Padded original sequences
        self.mask = np.asarray(padded_mask)     # Mask for original sequences


    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            # Shuffle raw_inputs, targets, and consequently self.inputs and self.mask if they are to be used directly
            self.raw_inputs = [self.raw_inputs[i] for i in shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            # Re-pad if necessary, or ensure get_slice uses shuffled raw_inputs
            current_padded_inputs, current_padded_mask, _ = data_masks(self.raw_inputs, [0])
            self.inputs = np.asarray(current_padded_inputs)
            self.mask = np.asarray(current_padded_mask)

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def _augment_sequence_item_dropout(self, sequence, item_drop_prob):
        if item_drop_prob == 0.0:
            return list(sequence) # Return a copy

        # Determine original length before padding
        original_len = 0
        for item in sequence:
            if item != 0:
                original_len += 1
            else:
                break
        
        if original_len <= 1: # Cannot augment if too short
            return list(sequence)

        augmented_seq = []
        for i in range(original_len):
            if torch.rand(1).item() >= item_drop_prob:
                augmented_seq.append(sequence[i])
        
        if not augmented_seq: # Ensure at least one item remains
            augmented_seq.append(sequence[0])
            
        # Pad with 0s to maintain original sequence length (self.len_max for the batch)
        # The sequence passed here is already padded to self.len_max by self.inputs[i]
        # So, we need to fill the rest with 0s up to that length
        padding_count = len(sequence) - len(augmented_seq)
        augmented_seq.extend([0] * padding_count)
        return augmented_seq

    def _get_graph_data_for_view(self, current_inputs_batch, current_len_max):
        # current_inputs_batch: list of sequences for the batch, each sequence is a list of item IDs
        # current_len_max: max sequence length for this batch (already padded to this)
        
        items_list, n_node_list, A_list, alias_inputs_list = [], [], [], []
        mask_list = [] # Also generate mask for this specific view

        batch_max_n_node = 0
        temp_n_nodes_for_batch = []
        for u_input_single in current_inputs_batch:
            # Consider only non-zero items for unique node count
            unique_nodes_in_seq = np.unique(np.array(u_input_single)[np.array(u_input_single) != 0])
            num_unique = len(unique_nodes_in_seq)
            temp_n_nodes_for_batch.append(num_unique if num_unique > 0 else 1) # Avoid 0 for max calc
        
        if temp_n_nodes_for_batch:
            batch_max_n_node = np.max(temp_n_nodes_for_batch)
        else: # Should not happen if batch is not empty
            batch_max_n_node = 1


        for u_input_single in current_inputs_batch: # u_input_single is a list of item IDs
            mask_list.append([1 if item != 0 else 0 for item in u_input_single])

            node = np.unique(np.array(u_input_single)[np.array(u_input_single) != 0])
            if len(node) == 0: # If sequence became all zeros after augmentation
                node = np.array([0]) # Placeholder unique node (will map to zero embedding typically)
            
            items_list.append(node.tolist() + (batch_max_n_node - len(node)) * [0])
            
            u_A = np.zeros((batch_max_n_node, batch_max_n_node))
            # Map item IDs to their 0-indexed position in the 'node' list for this sequence
            item_to_idx_map = {item_id: k for k, item_id in enumerate(node)}

            for i_idx in np.arange(len(u_input_single) - 1):
                item_curr = u_input_single[i_idx]
                item_next = u_input_single[i_idx+1]

                if item_next == 0: # End of effective sequence (due to padding or augmentation)
                    break 
                if item_curr == 0: # Current item is padding
                    continue
                
                # Check if items are in the current unique 'node' list for this sequence
                if item_curr in item_to_idx_map and item_next in item_to_idx_map:
                    u = item_to_idx_map[item_curr]
                    v = item_to_idx_map[item_next]
                    u_A[u][v] = 1
            
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1 # Avoid division by zero
            u_A_in = np.divide(u_A, u_sum_in)
            
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1 # Avoid division by zero
            u_A_out = np.divide(u_A.transpose(), u_sum_out) # Transpose before dividing for out-degree
            
            A_list.append(np.concatenate([u_A_in, u_A_out]).transpose())

            # Create alias_inputs: map items in original sequence positions to their indices in 'node' list
            alias_for_current_seq = []
            for item_val_in_seq in u_input_single:
                if item_val_in_seq == 0: # Is padding
                    alias_for_current_seq.append(0) # Assuming 0 index in 'node' can be a padding node or handled
                elif item_val_in_seq in item_to_idx_map:
                    alias_for_current_seq.append(item_to_idx_map[item_val_in_seq])
                else: # Should not happen if item_to_idx_map is correct for non-zero items
                    alias_for_current_seq.append(0) 
            alias_inputs_list.append(alias_for_current_seq)
            
        return np.array(alias_inputs_list), np.array(A_list), np.array(items_list), np.array(mask_list)

    def get_slice(self, i, ssl_item_drop_prob=0.2):
        # i here is a slice of indices for the batch
        batch_raw_inputs = [self.raw_inputs[idx] for idx in i]
        batch_targets = self.targets[i]
        # Original mask for the main task (based on unaugmented raw inputs)
        _, batch_mask_main, len_max_for_batch = data_masks(batch_raw_inputs, [0])


        # Create View 1 (e.g., less or no augmentation, used for main task and SSL)
        # For simplicity, let's use no augmentation for view 1 if ssl_item_drop_prob is for view 2.
        # Or, view 1 could also be augmented with a *different* small probability.
        # Here, view 1 = original (effectively item_drop_prob=0)
        inputs_v1_unpadded = [self._augment_sequence_item_dropout(seq, 0.0) for seq in batch_raw_inputs]
         # Pad them to len_max_for_batch
        inputs_v1_padded, _, _ = data_masks(inputs_v1_unpadded, [0])


        # Create View 2 (augmented, for SSL)
        inputs_v2_unpadded = [self._augment_sequence_item_dropout(seq, ssl_item_drop_prob) for seq in batch_raw_inputs]
        # Pad them to len_max_for_batch
        inputs_v2_padded, _, _ = data_masks(inputs_v2_unpadded, [0])


        alias_v1, A_v1, items_v1, mask_v1_ssl = self._get_graph_data_for_view(inputs_v1_padded, len_max_for_batch)
        alias_v2, A_v2, items_v2, mask_v2_ssl = self._get_graph_data_for_view(inputs_v2_padded, len_max_for_batch)

        # data_v1: data for the first view (used for main task and as one side of SSL)
        # data_v2: data for the second view (used as other side of SSL)
        # batch_targets: original targets for the main task
        # batch_mask_main: original mask for the main task scoring (from un-augmented data)
        return (alias_v1, A_v1, items_v1, mask_v1_ssl), \
               (alias_v2, A_v2, items_v2, mask_v2_ssl), \
               batch_targets, np.array(batch_mask_main)
