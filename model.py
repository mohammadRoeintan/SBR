############################################################
# This code builds on https://github.com/CRIPAC-DIG/TAGNN #
############################################################

from tqdm import tqdm
import datetime
import math
import numpy as np

import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class Attention_GNN(Module):
    def __init__(self, hidden_size, step=2):
        super(Attention_GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        # اضافه کردن dropout
        self.dropout = nn.Dropout(0.2)

        self.reset_parameters() # Initialize parameters

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                 weight.data.uniform_(-stdv, stdv)
            else:
                 weight.data.uniform_(-stdv, stdv)
        for layer in [self.linear_edge_in, self.linear_edge_out, self.linear_edge_f]:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data.uniform_(-stdv, stdv)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        batch_size, num_nodes_in_hidden, _ = hidden.shape
        expected_A_feature_dim = 2 * num_nodes_in_hidden

        # Ensure A has the correct feature dimension
        if A.shape[2] < expected_A_feature_dim:
            pad_size = expected_A_feature_dim - A.shape[2]
            pad_tensor = torch.zeros(batch_size, A.shape[1], pad_size,
                                    device=A.device, dtype=A.dtype)
            A_compat = torch.cat([A, pad_tensor], dim=2)
        elif A.shape[2] > expected_A_feature_dim:
            A_compat = A[:, :, :expected_A_feature_dim]
        else:
            A_compat = A

        input_in_adj = A_compat[:, :, :num_nodes_in_hidden]
        input_out_adj = A_compat[:, :, num_nodes_in_hidden:2*num_nodes_in_hidden]

        input_in = torch.matmul(input_in_adj, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(input_out_adj, self.linear_edge_out(hidden)) + self.b_oah

        input_in = self.dropout(input_in)
        input_out = self.dropout(input_out)

        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class Attention_SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(Attention_SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.nonhybrid = opt.nonhybrid

        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)

        self.max_len = opt.max_len
        self.position_emb_dim = opt.position_emb_dim
        if self.position_emb_dim != self.hidden_size:
            self.position_proj = nn.Linear(self.hidden_size + self.position_emb_dim, self.hidden_size)
            self.requires_projection = True
        else:
            self.requires_projection = False
            self.position_proj = None

        self.position_embedding = nn.Embedding(self.max_len + 1, self.position_emb_dim, padding_idx=0)

        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, projection_dim)
        )
        self.reset_parameters()

        self.dropout = nn.Dropout(0.2)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name.startswith("tagnn."):
                continue
            if 'bias' in name:
                nn.init.zeros_(weight)
            else:
                if 'embedding.weight' in name and 'position_embedding' not in name:
                    nn.init.xavier_uniform_(weight)
                elif 'position_embedding.weight' in name:
                    nn.init.xavier_uniform_(weight)
                elif weight.dim() > 1:
                    weight.data.uniform_(-stdv, stdv)

        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        if self.position_embedding.padding_idx is not None:
            with torch.no_grad():
                self.position_embedding.weight[self.position_embedding.padding_idx].fill_(0)


    def _get_seq_hidden_with_position(self, gnn_output_on_unique_nodes, alias_inputs_for_sequence, position_ids_for_sequence):
        batch_size = gnn_output_on_unique_nodes.size(0)
        # max_seq_len_in_batch = alias_inputs_for_sequence.size(1) # Not explicitly used for output shape here

        alias_indices_expanded = alias_inputs_for_sequence.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)

        seq_item_hidden = torch.gather(
            gnn_output_on_unique_nodes,
            1,
            alias_indices_expanded
        ) # Shape: (batch_size, alias_inputs_for_sequence.size(1), hidden_size)

        pos_embeds = self.position_embedding(position_ids_for_sequence.long())
        # Ensure pos_embeds matches seq_item_hidden's sequence length if they are different
        # This might happen if alias_inputs_for_sequence.size(1) differs from position_ids_for_sequence.size(1)
        # or if position_ids go out of bounds for self.position_embedding (max_len related)
        # For now, assuming they are created consistently by _get_graph_data_for_view

        if pos_embeds.shape[1] != seq_item_hidden.shape[1]:
            # This indicates an inconsistency in how alias_inputs and position_ids were prepared.
            # This should ideally be fixed in data preparation (proc_utils.py).
            # As a temporary safeguard, truncate or pad pos_embeds to match seq_item_hidden.
            # print(f"Warning: Mismatch in seq lengths for item_hidden ({seq_item_hidden.shape[1]}) and pos_embeds ({pos_embeds.shape[1]})")
            target_len = seq_item_hidden.shape[1]
            if pos_embeds.shape[1] > target_len:
                pos_embeds = pos_embeds[:, :target_len, :]
            else: # pos_embeds.shape[1] < target_len
                padding_size = target_len - pos_embeds.shape[1]
                pad_tensor = torch.zeros(pos_embeds.shape[0], padding_size, pos_embeds.shape[2],
                                         device=pos_embeds.device, dtype=pos_embeds.dtype)
                pos_embeds = torch.cat([pos_embeds, pad_tensor], dim=1)


        if self.position_emb_dim == self.hidden_size:
            seq_hidden_final = seq_item_hidden + pos_embeds
        else:
            combined = torch.cat([seq_item_hidden, pos_embeds], dim=-1)
            seq_hidden_final = self.position_proj(combined)

        return seq_hidden_final


    def forward(self, unique_item_inputs, A_matrix):
        hidden = self.embedding(unique_item_inputs)
        hidden = self.dropout(hidden)
        hidden = self.tagnn(A_matrix, hidden)

        transformer_key_padding_mask = (unique_item_inputs == self.embedding.padding_idx)

        x = hidden
        x_norm = self.layer_norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm,
                              key_padding_mask=transformer_key_padding_mask)
        hidden = x + self.dropout(x_attn)

        return hidden


    def compute_scores(self, seq_hidden_time_aware, mask_for_scoring):
        if seq_hidden_time_aware.size(0) == 0 : # No items in batch
            num_candidates = self.embedding.weight.size(0) -1
            return torch.empty(0, num_candidates, device=seq_hidden_time_aware.device)
        
        # If sequence length dimension is 0, but batch is not empty (should ideally not happen if data prep is correct)
        if seq_hidden_time_aware.size(1) == 0:
            # This means sequences are empty. ht will be problematic.
            # Alpha calculation will also be problematic.
            # Return empty scores for this batch.
            num_candidates = self.embedding.weight.size(0) -1
            return torch.empty(seq_hidden_time_aware.size(0), num_candidates, device=seq_hidden_time_aware.device)


        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        max_valid_index_for_ht = seq_hidden_time_aware.shape[1] - 1
        
        if max_valid_index_for_ht < 0: # Should be caught by seq_hidden_time_aware.size(1) == 0 check
             last_item_indices = torch.zeros_like(actual_lengths, device=seq_hidden_time_aware.device)
        else:
            last_item_indices = torch.clamp(actual_lengths - 1, min=0, max=max_valid_index_for_ht)

        batch_indices = torch.arange(seq_hidden_time_aware.shape[0], device=seq_hidden_time_aware.device)
        ht = seq_hidden_time_aware[batch_indices, last_item_indices]

        if ht.size(0) == 0 :
            num_candidates = self.embedding.weight.size(0) -1
            return torch.empty(0, num_candidates, device=seq_hidden_time_aware.device)

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])

        # --- START OF PROPOSED FIX FOR SIZEMISMATCH ---
        # Ensure q2 (derived from seq_hidden_time_aware) and mask_for_scoring have compatible sequence lengths
        
        data_seq_len = seq_hidden_time_aware.shape[1]  # e.g., 57 from error
        mask_seq_len = mask_for_scoring.shape[1]      # e.g., 50 from error

        if data_seq_len == mask_seq_len:
            seq_hidden_for_q2 = seq_hidden_time_aware
            effective_mask = mask_for_scoring
        elif data_seq_len > mask_seq_len:
            # Data is longer than mask. Truncate data. Mask is source of truth for length here.
            # print(f"Warning: compute_scores - Truncating data sequence from {data_seq_len} to {mask_seq_len} to match mask.")
            seq_hidden_for_q2 = seq_hidden_time_aware[:, :mask_seq_len, :]
            effective_mask = mask_for_scoring
        else: # data_seq_len < mask_seq_len
            # Mask is longer than data. This is more problematic.
            # It implies current_batch_max_len for mask was larger than for data.
            # Option 1: Truncate mask (assumes data length is the true limit for this operation)
            # print(f"Warning: compute_scores - Truncating mask sequence from {mask_seq_len} to {data_seq_len} to match data.")
            effective_mask = mask_for_scoring[:, :data_seq_len]
            seq_hidden_for_q2 = seq_hidden_time_aware
            # Option 2: Pad data (less safe as it introduces zeros not originally there)
            # padding_size = mask_seq_len - data_seq_len
            # pad_tensor = torch.zeros(seq_hidden_time_aware.shape[0], padding_size, seq_hidden_time_aware.shape[2],
            # device=seq_hidden_time_aware.device, dtype=seq_hidden_time_aware.dtype)
            # seq_hidden_for_q2 = torch.cat([seq_hidden_time_aware, pad_tensor], dim=1)
            # effective_mask = mask_for_scoring
            #
            # For now, choosing to truncate the mask to match data if data is shorter.
            # This means attention will only be over the available data length.
            # Ideally, data_seq_len and mask_seq_len should match from data loading.
            # If this branch is hit often, it indicates a deeper issue in data prep.

        q2 = self.linear_two(seq_hidden_for_q2) # q2 now has seq_len = effective_mask.shape[1]
        
        # q1 is (B, 1, H), q2 is (B, effective_mask.shape[1], H)
        # After broadcasting q1, (q1+q2) will be (B, effective_mask.shape[1], H)
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)) # alpha_logits is (B, effective_mask.shape[1], 1)

        # Now, effective_mask should be used for masked_fill
        alpha_logits_masked = alpha_logits.masked_fill(effective_mask.unsqueeze(-1) == 0, -1e9)
        # --- END OF PROPOSED FIX ---

        alpha = F.softmax(alpha_logits_masked, dim=1)

        # Use seq_hidden_for_q2 here as well, as its length matches alpha
        a = torch.sum(alpha * seq_hidden_for_q2 * effective_mask.unsqueeze(-1).float(), dim=1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], dim=1))

        candidate_item_embeddings = self.embedding.weight[1:]
        scores = torch.matmul(a, candidate_item_embeddings.transpose(0, 1))
        return scores

    def get_session_embedding_for_ssl(self, seq_hidden_time_aware, mask_for_ssl):
        if seq_hidden_time_aware.size(0) == 0 :
             # Fallback for projection_dim if projection_head_ssl is not fully initialized or accessible
             default_dim = self.hidden_size
             try:
                 default_dim = self.projection_head_ssl[-1].out_features
             except (AttributeError, IndexError, TypeError):
                 pass # Use self.hidden_size
             return torch.empty(0, default_dim, device=seq_hidden_time_aware.device)

        if seq_hidden_time_aware.size(1) == 0: # Zero sequence length
             default_dim = self.hidden_size
             try:
                 default_dim = self.projection_head_ssl[-1].out_features
             except (AttributeError, IndexError, TypeError):
                 pass
             return torch.empty(seq_hidden_time_aware.size(0), default_dim, device=seq_hidden_time_aware.device)


        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        max_valid_index_ssl = seq_hidden_time_aware.shape[1] - 1
        if max_valid_index_ssl < 0: # Should be caught by size(1)==0
            last_item_indices = torch.zeros_like(actual_lengths, device=seq_hidden_time_aware.device)
        else:
            last_item_indices = torch.clamp(actual_lengths - 1, min=0, max=max_valid_index_ssl)

        batch_indices = torch.arange(seq_hidden_time_aware.shape[0], device=seq_hidden_time_aware.device)
        session_repr = seq_hidden_time_aware[batch_indices, last_item_indices]
        return session_repr

    def calculate_infonce_loss(self, z1, z2):
        if z1.size(0) == 0 or z2.size(0) == 0 :
            # Determine device carefully if one tensor is empty
            dev = z1.device if z1.numel() > 0 else (z2.device if z2.numel() > 0 else torch.device("cpu"))
            return torch.tensor(0.0, device=dev, requires_grad=True)

        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        sim_matrix = torch.matmul(z1_norm, z2_norm.T) / self.ssl_temperature
        
        if sim_matrix.size(0) == 0:
             return torch.tensor(0.0, device=sim_matrix.device, requires_grad=True)
        labels = torch.arange(sim_matrix.size(0)).long().to(sim_matrix.device)
        loss_ssl = F.cross_entropy(sim_matrix, labels)
        return loss_ssl


def train_test(model, train_data, test_data, opt, device):
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

    print(f'Start training epoch {opt.current_epoch_num}: {datetime.datetime.now()}')
    model.train()

    total_loss_epoch = 0.0
    total_main_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    num_batches_processed = 0

    current_batch_size = opt.batchSize
    slices = train_data.generate_batch(current_batch_size)
    if not slices:
        print("Warning: No batches generated from training data. Skipping training for this epoch.")
        return 0.0, 0.0

    for i_slice_indices, j_batch_num in tqdm(zip(slices, np.arange(len(slices))), total=len(slices), desc=f"Epoch {opt.current_epoch_num} Training"):
        if len(i_slice_indices) == 0:
            continue

        model_module.optimizer.zero_grad()

        ssl_drop_prob = opt.ssl_item_drop_prob
        # Ensure get_slice returns a consistent structure even for empty batches
        # The original get_slice returns a tuple of (tuple, tuple, array, array)
        # If empty, it returns tuples of empty arrays.
        returned_data = train_data.get_slice(i_slice_indices, ssl_item_drop_prob=ssl_drop_prob)
        data_v1, data_v2, targets_main_np, mask_main_np = returned_data

        # Check if items_vX_unique (data_vX[2]) is empty or if data_vX is not as expected
        if not (isinstance(data_v1, tuple) and len(data_v1) == 5 and isinstance(data_v1[2], np.ndarray) and data_v1[2].size > 0):
            # print(f"Warning: Training - Empty or malformed data_v1 for slice {i_slice_indices}. Skipping batch.")
            continue
        if not (isinstance(data_v2, tuple) and len(data_v2) == 5 and isinstance(data_v2[2], np.ndarray) and data_v2[2].size > 0):
            # print(f"Warning: Training - Empty or malformed data_v2 for slice {i_slice_indices}. Skipping batch (SSL part might be affected).")
            # Depending on logic, may decide to proceed without SSL or skip entirely
            # For now, assuming if v2 is bad, we might still do main loss with v1 if v1 is good.
            # However, current code structure implies both are needed or none.
            # The original checks were data_v1[0].size == 0. item_unique is data_vX[2]
            pass # Let further checks handle items_vX_unique


        alias_inputs_v1, A_v1, items_v1_unique_np, mask_v1_ssl_np, position_ids_v1_np = data_v1
        alias_inputs_v2, A_v2, items_v2_unique_np, mask_v2_ssl_np, position_ids_v2_np = data_v2
        
        # Main loss part
        if items_v1_unique_np.size == 0 or targets_main_np.size == 0:
            # print(f"Warning: Training - items_v1_unique_np or targets_main_np is empty for slice {i_slice_indices}. Skipping main loss.")
            loss_main = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            items_v1_unique = torch.from_numpy(items_v1_unique_np).long().to(device)
            A_v1 = torch.from_numpy(A_v1).float().to(device)
            alias_inputs_v1 = torch.from_numpy(alias_inputs_v1).long().to(device)
            position_ids_v1 = torch.from_numpy(position_ids_v1_np).long().to(device)
            targets_main = torch.from_numpy(targets_main_np).long().to(device)
            mask_main = torch.from_numpy(mask_main_np).long().to(device)

            if items_v1_unique.size(0) == 0: # Should be caught by np check already
                loss_main = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                gnn_output_v1 = model(items_v1_unique, A_v1)
                if gnn_output_v1.size(0) == 0:
                    loss_main = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    seq_hidden_v1_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v1, alias_inputs_v1, position_ids_v1)
                    if seq_hidden_v1_time_aware.size(0) == 0 or seq_hidden_v1_time_aware.size(1) == 0 :
                        loss_main = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        scores_main = model_module.compute_scores(seq_hidden_v1_time_aware, mask_main)
                        if scores_main.size(0) == 0 and targets_main.size(0) == 0:
                             loss_main = torch.tensor(0.0, device=device, requires_grad=True)
                        elif scores_main.size(0) != targets_main.size(0): # Mismatch after compute_scores handling
                             # print(f"Error: Training - scores_main batch size ({scores_main.size(0)}) != targets_main batch size ({targets_main.size(0)}) for slice {i_slice_indices}.")
                             loss_main = torch.tensor(0.0, device=device, requires_grad=True) # Skip this batch's main loss
                        else:
                             loss_main = model_module.loss_function(scores_main, targets_main - 1)
        
        # SSL loss part
        # Re-fetch seq_hidden_v1_time_aware if it was skipped or modified for main loss due to empty items
        # This assumes data_v1 and data_v2 are independent enough that one can be processed if other fails.
        # Or, if main_loss path was skipped due to empty items_v1, seq_hidden_v1 might not be available.
        # For SSL, we need both v1 and v2 representations.
        
        if items_v1_unique_np.size == 0 or items_v2_unique_np.size == 0 or \
           mask_v1_ssl_np.size == 0 or mask_v2_ssl_np.size == 0 : # Check SSL specific masks too
            # print(f"Warning: Training - Data for SSL is incomplete for slice {i_slice_indices}. Skipping SSL loss.")
            loss_ssl = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Ensure items_v1_unique and related tensors are on device if not processed for main loss
            items_v1_unique = torch.from_numpy(items_v1_unique_np).long().to(device)
            A_v1_ssl = torch.from_numpy(A_v1).float().to(device) # A_v1 might be different from main if data_v1 modified
            alias_inputs_v1_ssl = torch.from_numpy(alias_inputs_v1_np).long().to(device)
            position_ids_v1_ssl = torch.from_numpy(position_ids_v1_np).long().to(device)
            mask_v1_ssl = torch.from_numpy(mask_v1_ssl_np).long().to(device)

            items_v2_unique = torch.from_numpy(items_v2_unique_np).long().to(device)
            A_v2 = torch.from_numpy(A_v2).float().to(device)
            alias_inputs_v2 = torch.from_numpy(alias_inputs_v2_np).long().to(device)
            position_ids_v2 = torch.from_numpy(position_ids_v2_np).long().to(device)
            mask_v2_ssl = torch.from_numpy(mask_v2_ssl_np).long().to(device)

            if items_v1_unique.size(0) == 0 or items_v2_unique.size(0) == 0: # Redundant check, but safe
                loss_ssl = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                gnn_output_v1_ssl = model(items_v1_unique, A_v1_ssl) # Recompute if necessary
                seq_hidden_v1_ssl_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v1_ssl, alias_inputs_v1_ssl, position_ids_v1_ssl)
                session_emb_v1_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v1_ssl_time_aware, mask_v1_ssl)

                gnn_output_v2 = model(items_v2_unique, A_v2)
                if gnn_output_v2.size(0) == 0 or session_emb_v1_ssl.size(0) == 0: # If gnn_output_v2 or emb_v1 is empty
                    loss_ssl = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    seq_hidden_v2_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v2, alias_inputs_v2, position_ids_v2)
                    session_emb_v2_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v2_time_aware, mask_v2_ssl)

                    if session_emb_v1_ssl.size(0) == 0 or session_emb_v2_ssl.size(0) == 0:
                        loss_ssl = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        projected_emb_v1 = model_module.projection_head_ssl(session_emb_v1_ssl)
                        projected_emb_v2 = model_module.projection_head_ssl(session_emb_v2_ssl)
                        loss_ssl = model_module.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)

        combined_loss = loss_main + model_module.ssl_weight * loss_ssl
        
        if combined_loss.requires_grad and (loss_main.requires_grad or loss_ssl.requires_grad): # Ensure there's something to backprop
            # Check if parameters have grads (they should if model ran)
            if any(p.grad is not None for p in model.parameters()): #This check is after backward, so not useful here
                 pass # Grads would exist if backward was already called

            if sum(p.numel() for p in model.parameters() if p.requires_grad) > 0: # Check if model has trainable params
                # Clip grads only if they are computed.
                # combined_loss.backward() computes them.
                # We need to ensure that backward is only called if combined_loss is not a zero tensor from skipped ops
                # and actually has a graph.
                
                # A simple check if loss is non-zero or has graph
                is_meaningful_loss = combined_loss.grad_fn is not None
                if not is_meaningful_loss and combined_loss.item() == 0.0: # check if it's just a zero tensor
                    pass # No backward pass if loss is effectively zero and has no graph
                else:
                    combined_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    model_module.optimizer.step()
            
        total_loss_epoch += combined_loss.item()
        total_main_loss_epoch += loss_main.item()
        total_ssl_loss_epoch += loss_ssl.item()
        num_batches_processed += 1

        if num_batches_processed > 0 and j_batch_num > 0 and len(slices) > 5 and j_batch_num % int(len(slices) / 5) == 0:
            print('[%d/%d] Total Loss: %.4f (Main: %.4f, SSL: %.4f)' %
                  (j_batch_num, len(slices), combined_loss.item(), loss_main.item(), loss_ssl.item()))

    if num_batches_processed > 0:
        model_module.scheduler.step()
        avg_total_loss = total_loss_epoch / num_batches_processed
        avg_main_loss = total_main_loss_epoch / num_batches_processed
        avg_ssl_loss = total_ssl_loss_epoch / num_batches_processed
        print(f'\tTraining Epoch {opt.current_epoch_num} Loss (Avg): Total: {avg_total_loss:.3f} (Main: {avg_main_loss:.3f}, SSL: {avg_ssl_loss:.3f})')
    else:
        print(f'\tNo batches were effectively processed in epoch {opt.current_epoch_num}.')

    print(f'Start Prediction for epoch {opt.current_epoch_num}: {datetime.datetime.now()}')
    model.eval()
    hit, mrr = [], []
    slices_test = test_data.generate_batch(opt.batchSize)
    if not slices_test:
        print("Warning: No batches generated from test data. Skipping evaluation for this epoch.")
        return 0.0, 0.0

    with torch.no_grad():
        for i_test_slice_indices in tqdm(slices_test, desc=f"Epoch {opt.current_epoch_num} Evaluation"):
            if len(i_test_slice_indices) == 0:
                continue

            returned_data_test = test_data.get_slice(i_test_slice_indices, ssl_item_drop_prob=0.0)
            data_v1_test, _, targets_test_orig_np, mask_test_orig_np = returned_data_test

            if not (isinstance(data_v1_test, tuple) and len(data_v1_test) == 5 and isinstance(data_v1_test[2], np.ndarray) and data_v1_test[2].size > 0):
                # print(f"Warning: Evaluation - Empty or malformed data_v1_test for slice {i_test_slice_indices}. Skipping.")
                continue

            alias_inputs_eval_np, A_eval_np, items_eval_unique_np, _, position_ids_eval_np = data_v1_test

            if items_eval_unique_np.size == 0:
                # print(f"Warning: Evaluation - items_eval_unique_np is empty for slice {i_test_slice_indices}. Skipping.")
                continue

            items_eval_unique = torch.from_numpy(items_eval_unique_np).long().to(device)
            A_eval = torch.from_numpy(A_eval_np).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval_np).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval_np).long().to(device)
            mask_test_orig_cuda = torch.from_numpy(mask_test_orig_np).long().to(device)
            targets_test_orig_cuda = torch.from_numpy(targets_test_orig_np).long().to(device)


            gnn_output_eval = model(items_eval_unique, A_eval)
            if gnn_output_eval.size(0) == 0:
                continue

            seq_hidden_eval_time_aware = model_module._get_seq_hidden_with_position(gnn_output_eval, alias_inputs_eval, position_ids_eval)
            if seq_hidden_eval_time_aware.size(0) == 0 or seq_hidden_eval_time_aware.size(1) == 0:
                continue

            scores_eval = model_module.compute_scores(seq_hidden_eval_time_aware, mask_test_orig_cuda)
            if scores_eval.size(0) == 0 or scores_eval.size(1) == 0: # Check if scores are empty or have no candidates
                 continue

            # Ensure k in topk is not greater than number of available items
            k_topk = min(20, scores_eval.size(1))
            if k_topk <=0 : # if no candidates to pick from
                continue
            sub_scores_top20_indices = scores_eval.topk(k_topk)[1]

            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            targets_eval_np = targets_test_orig_cuda.cpu().detach().numpy() - 1

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_np):
                hit.append(np.isin(target_item, score_row))
                if target_item in score_row:
                    mrr.append(1 / (np.where(score_row == target_item)[0][0] + 1))
                else:
                    mrr.append(0)

    hit_metric = np.mean(hit) * 100 if hit else 0.0
    mrr_metric = np.mean(mrr) * 100 if mrr else 0.0
    return hit_metric, mrr_metric
