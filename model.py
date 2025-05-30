############################################################
# This code builds on https://github.com/CRIPAC-DIG/TAGNN #
############################################################

from tqdm import tqdm
import datetime
import math
import numpy as np
import random

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
        self.linear_edge_f = nn.Linear( # This seems unused, but kept from original
            self.hidden_size, self.hidden_size, bias=True)
        
        # اضافه کردن dropout
        self.dropout = nn.Dropout(0.2)
        
        self.reset_parameters() # Initialize parameters

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1: # For weight matrices (w_ih, w_hh)
                 weight.data.uniform_(-stdv, stdv)
            else: # For biases (b_ih, b_hh, b_iah, b_oah)
                 # nn.init.zeros_(weight) # A common choice for biases
                 weight.data.uniform_(-stdv, stdv) # Or initialize like weights, as in original TAGNN GNNCell
        # Initialize linear layer weights/biases as well
        for layer in [self.linear_edge_in, self.linear_edge_out, self.linear_edge_f]:
            layer.weight.data.uniform_(-stdv, stdv)
            if layer.bias is not None:
                layer.bias.data.uniform_(-stdv, stdv) # Or zeros_

    def GNNCell(self, A, hidden):
        # A: Adjacency matrix, expected shape [batch, N, 2*N] where N is num_nodes
        # hidden: Node features, expected shape [batch, N, hidden_size]
        num_nodes_in_hidden = hidden.shape[1]

        # Initialize adj matrices to ensure they are always defined
        input_in_adj = None
        input_out_adj = None

        # Default assumption for A's shape based on hidden's current node dimension
        expected_A_node_dim = num_nodes_in_hidden
        expected_A_feature_dim = 2 * num_nodes_in_hidden
        
        current_A_node_dim = A.shape[1]
        current_A_feature_dim = A.shape[2]

        # Scenario 1: A matches hidden's node dimension perfectly
        if current_A_node_dim == expected_A_node_dim and current_A_feature_dim == expected_A_feature_dim:
            input_in_adj = A[:, :, :expected_A_node_dim]
            input_out_adj = A[:, :, expected_A_node_dim:expected_A_feature_dim]
        # Scenario 2: A's node dimension is larger (e.g., padded A for batch)
        # but its feature dimension is correctly 2 * its own node dimension
        elif current_A_node_dim > expected_A_node_dim and current_A_feature_dim == 2 * current_A_node_dim :
            # Slice A to match hidden's node dimension.
            # This implies A was constructed for a potentially larger max_nodes_in_batch
            # print(f"GNNCell: Slicing A's node dimension from {current_A_node_dim} to {expected_A_node_dim}")
            A_temp_sliced_nodes = A[:, :expected_A_node_dim, :]
            # Now, A_temp_sliced_nodes should be [batch, expected_A_node_dim, 2 * current_A_node_dim]
            # We need to ensure the feature dimension is also sliced if it was based on the larger current_A_node_dim
            if A_temp_sliced_nodes.shape[2] == 2 * current_A_node_dim : # if feature dim was based on original A node count
                input_in_adj = A_temp_sliced_nodes[:, :, :expected_A_node_dim]
                input_out_adj = A_temp_sliced_nodes[:, :, expected_A_node_dim:expected_A_feature_dim] # Use expected_A_feature_dim for slicing end
            elif A_temp_sliced_nodes.shape[2] == expected_A_feature_dim: # if feature dim already matches target hidden node count
                input_in_adj = A_temp_sliced_nodes[:, :, :expected_A_node_dim]
                input_out_adj = A_temp_sliced_nodes[:, :, expected_A_node_dim:expected_A_feature_dim]
            else:
                print(f"Warning: GNNCell - A (node-sliced) shape {A_temp_sliced_nodes.shape} feature dimension mismatch "
                      f"with expected_A_feature_dim {expected_A_feature_dim}. Fallback slicing.")
                # Fallback if feature dimension is also off after node slicing.
                input_in_adj = A_temp_sliced_nodes[:, :, :expected_A_node_dim]
                # Try to get the second half, ensuring it doesn't exceed bounds and has correct final dim
                slice_end_out = min(A_temp_sliced_nodes.shape[2], expected_A_node_dim + expected_A_node_dim)
                input_out_adj_raw = A_temp_sliced_nodes[:, :, expected_A_node_dim:slice_end_out]
                # Ensure input_out_adj has last dimension == expected_A_node_dim
                if input_out_adj_raw.shape[2] < expected_A_node_dim:
                    padding_needed = expected_A_node_dim - input_out_adj_raw.shape[2]
                    pad_tensor = torch.zeros(*input_out_adj_raw.shape[:2], padding_needed, device=A.device, dtype=A.dtype)
                    input_out_adj = torch.cat([input_out_adj_raw, pad_tensor], dim=2)
                else:
                    input_out_adj = input_out_adj_raw[:,:,:expected_A_node_dim]


        # Scenario 3: A's dimensions are problematic (fallback / warning)
        else:
            print(f"Warning: GNNCell - A shape {A.shape} does not align well with hidden shape {hidden.shape} "
                  f"(expected A: [*, {expected_A_node_dim}, {expected_A_feature_dim}]). "
                  f"Attempting robust slicing. Results might be incorrect.")
            
            # Try to make the best guess for slicing
            # Slice node dimension of A to match hidden, if A is larger
            A_compat = A
            if A.shape[1] > expected_A_node_dim:
                A_compat = A[:, :expected_A_node_dim, :]
            
            # Now A_compat has dim1 = expected_A_node_dim or A.shape[1] if it was smaller/equal
            # Its dim2 is A_compat.shape[2]
            
            # For input_in_adj, take the first expected_A_node_dim columns from A_compat's feature dimension
            input_in_adj = A_compat[:, :, :min(A_compat.shape[2], expected_A_node_dim)]
            if input_in_adj.shape[2] < expected_A_node_dim: # Pad if too short
                padding_needed = expected_A_node_dim - input_in_adj.shape[2]
                pad_tensor = torch.zeros(*input_in_adj.shape[:2], padding_needed, device=A.device, dtype=A.dtype)
                input_in_adj = torch.cat([input_in_adj, pad_tensor], dim=2)

            # For input_out_adj, try to take the next expected_A_node_dim columns
            if A_compat.shape[2] > expected_A_node_dim:
                input_out_adj = A_compat[:, :, expected_A_node_dim:min(A_compat.shape[2], 2 * expected_A_node_dim)]
                if input_out_adj.shape[2] < expected_A_node_dim: # Pad if too short
                    padding_needed = expected_A_node_dim - input_out_adj.shape[2]
                    pad_tensor = torch.zeros(*input_out_adj.shape[:2], padding_needed, device=A.device, dtype=A.dtype)
                    input_out_adj = torch.cat([input_out_adj, pad_tensor], dim=2)
                elif input_out_adj.shape[2] > expected_A_node_dim: # Truncate if too long
                     input_out_adj = input_out_adj[:,:,:expected_A_node_dim]

            else: # If A_compat.shape[2] is not even > expected_A_node_dim, out_adj will be problematic
                print(f"ERROR: GNNCell - Cannot form input_out_adj from A_compat shape {A_compat.shape}")
                # Create a zero tensor of the expected shape to prevent matmul error, but this is bad.
                input_out_adj = torch.zeros(A_compat.shape[0], A_compat.shape[1], expected_A_node_dim, device=A.device, dtype=A.dtype)

        # Final check if variables are defined (should be by now)
        if input_in_adj is None or input_out_adj is None:
            print("FATAL: GNNCell - input_in_adj or input_out_adj not defined. This should not happen.")
            # Fallback to zero matrices to avoid runtime error, but this is a severe issue.
            input_in_adj = torch.zeros(hidden.shape[0], num_nodes_in_hidden, num_nodes_in_hidden, device=A.device, dtype=A.dtype)
            input_out_adj = torch.zeros(hidden.shape[0], num_nodes_in_hidden, num_nodes_in_hidden, device=A.device, dtype=A.dtype)


        input_in = torch.matmul(input_in_adj, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(input_out_adj, self.linear_edge_out(hidden)) + self.b_oah

        # اعمال dropout
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
        # Ensure position_emb_dim is compatible for addition if that's the combination strategy
        if self.position_emb_dim != self.hidden_size:
            print(f"Model Init Warning: position_emb_dim ({self.position_emb_dim}) is not equal to hidden_size ({self.hidden_size}). "
                  "If combining by addition, this will cause errors. Ensure they are equal or implement projection.")
            # For this implementation, we assume they should be equal for addition in _get_seq_hidden_with_position
            # self.position_emb_dim = self.hidden_size # Or handle projection if different dims are intended
            
        self.position_embedding = nn.Embedding(self.max_len + 1, self.position_emb_dim, padding_idx=0)

        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        
        # افزایش تعداد attention heads و اضافه کردن dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,  # افزایش از 2 به 4
            dropout=0.2,
            batch_first=True
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # اضافه کردن label smoothing به loss function
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, projection_dim)
        )
        self.reset_parameters() # Call after all layers are defined
        
        # اضافه کردن dropout
        self.dropout = nn.Dropout(0.2)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            # Check if parameter belongs to the GNN submodule and skip if GNN handles its own init
            if name.startswith("tagnn."): 
                continue # Assuming Attention_GNN handles its own parameter initialization
            
            if 'bias' in name:
                nn.init.zeros_(weight)
            else:
                if 'embedding.weight' in name and not name.startswith("position_embedding"): # Item embedding
                    nn.init.xavier_uniform_(weight)
                elif 'position_embedding.weight' in name: # Position embedding
                    nn.init.xavier_uniform_(weight)
                elif weight.dim() > 1 : # For other linear layers' weights etc.
                    weight.data.uniform_(-stdv, stdv)
        
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        if self.position_embedding.padding_idx is not None:
            with torch.no_grad():
                self.position_embedding.weight[self.position_embedding.padding_idx].fill_(0)


    def _get_seq_hidden_with_position(self, gnn_output_on_unique_nodes, alias_inputs_for_sequence, position_ids_for_sequence):
        batch_size = gnn_output_on_unique_nodes.size(0)
        max_seq_len_in_batch = alias_inputs_for_sequence.size(1) # Max length of sequences in this batch
        
        # alias_inputs_for_sequence: [batch, max_seq_len_in_batch]
        # gnn_output_on_unique_nodes: [batch, max_unique_nodes_in_batch, hidden_size]
        # We need to gather along dim 1 of gnn_output
        
        # Ensure alias_inputs are long and correctly shaped for gather/indexing
        # It should be [batch, max_seq_len_in_batch] with indices up to max_unique_nodes_in_batch - 1
        alias_indices_expanded = alias_inputs_for_sequence.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        # Perform gather operation
        try:
            seq_item_hidden = torch.gather(gnn_output_on_unique_nodes, 1, alias_indices_expanded)
        except RuntimeError as e:
            print(f"Error during torch.gather in _get_seq_hidden_with_position:")
            print(f"  gnn_output_on_unique_nodes shape: {gnn_output_on_unique_nodes.shape}")
            print(f"  alias_inputs_for_sequence shape: {alias_inputs_for_sequence.shape}")
            print(f"  alias_indices_expanded shape: {alias_indices_expanded.shape}")
            print(f"  Max index in alias_inputs: {alias_inputs_for_sequence.max()}")
            raise e
            
        pos_embeds = self.position_embedding(position_ids_for_sequence.long())
        
        if self.position_emb_dim == self.hidden_size:
            seq_hidden_final = seq_item_hidden + pos_embeds
        else:
            print(f"Critical Warning in _get_seq_hidden_with_position: position_emb_dim ({self.position_emb_dim}) != hidden_size ({self.hidden_size}). "
                  "Positional embeddings not added as intended. Fix dimensions or implement projection.")
            seq_hidden_final = seq_item_hidden # Fallback to prevent dimension mismatch error with addition
            
        return seq_hidden_final


    def forward(self, unique_item_inputs, A_matrix):
        hidden = self.embedding(unique_item_inputs) # [batch, max_unique_nodes, hidden_size]
        hidden = self.dropout(hidden)  # اضافه کردن dropout
        hidden = self.tagnn(A_matrix, hidden)       # [batch, max_unique_nodes, hidden_size] (after GNN)
        
        transformer_key_padding_mask = (unique_item_inputs == self.embedding.padding_idx) # [N, S_unique]
        
        # Corrected Transformer block application
        x = hidden # from GNN
        x_norm = self.layer_norm1(x) # Pre-LN
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=transformer_key_padding_mask)
        hidden = x + self.dropout(x_attn)  # اضافه کردن dropout به خروجی attention
        
        return hidden


    def compute_scores(self, seq_hidden_time_aware, mask_for_scoring):
        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        # Handle cases where actual_lengths might be 0 (e.g., if a sequence in batch is all padding)
        # last_item_indices should be at least 0.
        last_item_indices = torch.max(torch.zeros_like(actual_lengths, device=actual_lengths.device), actual_lengths - 1)
        
        # Gather ht safely. If a row in seq_hidden_time_aware has 0 actual length,
        # last_item_indices will be 0, so ht will be the first (likely padding) embedding.
        ht = seq_hidden_time_aware[torch.arange(seq_hidden_time_aware.shape[0], device=seq_hidden_time_aware.device).long(), last_item_indices]

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_hidden_time_aware)
        
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2))
        # Apply mask before softmax: set logits for padded positions to a very small number
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -1e9) # Or -torch.finfo(alpha_logits.dtype).max
        alpha = F.softmax(alpha_logits_masked, dim=1)
        
        # Weighted sum of hidden states, ensure only actual items contribute via mask
        a = torch.sum(alpha * seq_hidden_time_aware * mask_for_scoring.unsqueeze(-1).float(), dim=1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], dim=1))

        # Candidate item embeddings (all items except padding idx 0, if 0 is used for padding)
        candidate_item_embeddings = self.embedding.weight[1:] 
        scores = torch.matmul(a, candidate_item_embeddings.transpose(0, 1))
        return scores

    def get_session_embedding_for_ssl(self, seq_hidden_time_aware, mask_for_ssl):
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.max(torch.zeros_like(actual_lengths, device=actual_lengths.device), actual_lengths - 1)
        session_repr = seq_hidden_time_aware[torch.arange(seq_hidden_time_aware.shape[0], device=seq_hidden_time_aware.device).long(), last_item_indices]
        return session_repr

    def calculate_infonce_loss(self, z1, z2):
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        sim_matrix = torch.matmul(z1_norm, z2_norm.T) / self.ssl_temperature
        labels = torch.arange(z1_norm.size(0)).long().to(z1_norm.device)
        loss_ssl = F.cross_entropy(sim_matrix, labels)
        return loss_ssl


def train_test(model, train_data, test_data, opt, device):
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # Ensure scheduler step is called only once per epoch effectively
    # model_module.scheduler.last_epoch starts at -1. opt.epoch is total epochs.
    # Current epoch can be inferred if train_test is called in a loop or taken as arg.
    # Assuming train_test is called for one epoch of training:
    if model_module.scheduler.last_epoch < opt.current_epoch_num : # Pass current_epoch_num to train_test
         model_module.scheduler.step()

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
        if len(i_slice_indices) == 0: continue

        model_module.optimizer.zero_grad()

        ssl_drop_prob = opt.ssl_item_drop_prob
        data_v1, data_v2, targets_main_np, mask_main_np = train_data.get_slice(i_slice_indices, ssl_item_drop_prob=ssl_drop_prob)

        if data_v1[0].size == 0 or data_v2[0].size == 0:
            # print(f"Skipping empty batch {j_batch_num} in epoch {opt.current_epoch_num}")
            continue

        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl, position_ids_v1 = data_v1
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl, position_ids_v2 = data_v2

        # --- Move data to device ---
        items_v1_unique = torch.from_numpy(items_v1_unique).long().to(device)
        A_v1 = torch.from_numpy(A_v1).float().to(device)
        alias_inputs_v1 = torch.from_numpy(alias_inputs_v1).long().to(device)
        mask_v1_ssl = torch.from_numpy(mask_v1_ssl).long().to(device)
        position_ids_v1 = torch.from_numpy(position_ids_v1).long().to(device)

        items_v2_unique = torch.from_numpy(items_v2_unique).long().to(device)
        A_v2 = torch.from_numpy(A_v2).float().to(device)
        alias_inputs_v2 = torch.from_numpy(alias_inputs_v2).long().to(device)
        mask_v2_ssl = torch.from_numpy(mask_v2_ssl).long().to(device)
        position_ids_v2 = torch.from_numpy(position_ids_v2).long().to(device)

        targets_main = torch.from_numpy(targets_main_np).long().to(device)
        mask_main = torch.from_numpy(mask_main_np).long().to(device)
        # --- End Move data to device ---

        # --- Main Task (using view 1) ---
        gnn_output_v1 = model(items_v1_unique, A_v1)
        seq_hidden_v1_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v1, alias_inputs_v1, position_ids_v1)
        
        # Check for empty sequences after processing, before scoring
        if seq_hidden_v1_time_aware.size(0) == 0 : # Batch dimension is 0
            # print(f"Skipping batch {j_batch_num} due to empty seq_hidden_v1_time_aware.")
            continue
            
        scores_main = model_module.compute_scores(seq_hidden_v1_time_aware, mask_main)
        loss_main = model_module.loss_function(scores_main, targets_main - 1) # targets are 1-indexed

        # --- SSL Task ---
        session_emb_v1_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v1_time_aware, mask_v1_ssl)
        
        gnn_output_v2 = model(items_v2_unique, A_v2)
        seq_hidden_v2_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v2, alias_inputs_v2, position_ids_v2)
        if seq_hidden_v2_time_aware.size(0) == 0 :
            # print(f"Skipping batch {j_batch_num} due to empty seq_hidden_v2_time_aware.")
            continue # Should not happen if v1 was not empty
        session_emb_v2_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v2_time_aware, mask_v2_ssl)

        projected_emb_v1 = model_module.projection_head_ssl(session_emb_v1_ssl)
        projected_emb_v2 = model_module.projection_head_ssl(session_emb_v2_ssl)
        loss_ssl = model_module.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)

        combined_loss = loss_main + model_module.ssl_weight * loss_ssl
        
        # اضافه کردن gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        combined_loss.backward()
        model_module.optimizer.step()

        total_loss_epoch += combined_loss.item()
        total_main_loss_epoch += loss_main.item()
        total_ssl_loss_epoch += loss_ssl.item()
        num_batches_processed += 1

        if num_batches_processed > 0 and j_batch_num > 0 and len(slices) > 5 and j_batch_num % int(len(slices) / 5) == 0 :
            print('[%d/%d] Total Loss: %.4f (Main: %.4f, SSL: %.4f)' %
                  (j_batch_num, len(slices), combined_loss.item(), loss_main.item(), loss_ssl.item()))
    
    if num_batches_processed > 0:
        avg_total_loss = total_loss_epoch / num_batches_processed
        avg_main_loss = total_main_loss_epoch / num_batches_processed
        avg_ssl_loss = total_ssl_loss_epoch / num_batches_processed
        print(f'\tTraining Epoch {opt.current_epoch_num} Loss (Avg): Total: {avg_total_loss:.3f} (Main: {avg_main_loss:.3f}, SSL: {avg_ssl_loss:.3f})')
    else:
        print(f'\tNo batches were effectively processed in epoch {opt.current_epoch_num}.')

    # --- Evaluation ---
    print(f'Start Prediction for epoch {opt.current_epoch_num}: {datetime.datetime.now()}')
    model.eval()
    hit, mrr = [], []
    slices_test = test_data.generate_batch(opt.batchSize)
    if not slices_test:
        print("Warning: No batches generated from test data. Skipping evaluation for this epoch.")
        return 0.0, 0.0

    with torch.no_grad():
        for i_test_slice_indices in tqdm(slices_test, desc=f"Epoch {opt.current_epoch_num} Evaluation"):
            if len(i_test_slice_indices) == 0: continue

            data_v1_test, _, targets_test_orig_np, mask_test_orig_np = test_data.get_slice(i_test_slice_indices, ssl_item_drop_prob=0.0)
            
            if data_v1_test[0].size == 0: continue

            alias_inputs_eval, A_eval, items_eval_unique, _, position_ids_eval = data_v1_test

            items_eval_unique = torch.from_numpy(items_eval_unique).long().to(device)
            A_eval = torch.from_numpy(A_eval).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval).long().to(device)
            mask_test_orig_cuda = torch.from_numpy(mask_test_orig_np).long().to(device)
            targets_test_orig_cuda = torch.from_numpy(targets_test_orig_np).long().to(device)

            gnn_output_eval = model(items_eval_unique, A_eval)
            if gnn_output_eval.size(0) == 0: continue

            seq_hidden_eval_time_aware = model_module._get_seq_hidden_with_position(gnn_output_eval, alias_inputs_eval, position_ids_eval)
            if seq_hidden_eval_time_aware.size(0) == 0: continue
                
            scores_eval = model_module.compute_scores(seq_hidden_eval_time_aware, mask_test_orig_cuda)

            sub_scores_top20_indices = scores_eval.topk(20)[1]
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            targets_eval_np = targets_test_orig_cuda.cpu().detach().numpy()

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_np):
                target_for_eval = target_item - 1 # targets are 1-indexed
                hit.append(np.isin(target_for_eval, score_row))
                if target_for_eval in score_row:
                    mrr.append(1 / (np.where(score_row == target_for_eval)[0][0] + 1))
                else:
                    mrr.append(0)

    hit_metric = np.mean(hit) * 100 if hit else 0.0
    mrr_metric = np.mean(mrr) * 100 if mrr else 0.0
    return hit_metric, mrr_metric
