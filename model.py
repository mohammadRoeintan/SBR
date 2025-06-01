from tqdm import tqdm
import datetime
import math
import numpy as np

import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

class TimeAwareStarGNN(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.time_embed = nn.Linear(1, hidden_size)

        self.star_center = nn.Parameter(torch.Tensor(hidden_size))
        stdv = 1.0 / math.sqrt(hidden_size)
        self.star_center.data.uniform_(-stdv, stdv)

        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, hidden_seq, time_diffs_seq):
        if hidden_seq.size(1) != time_diffs_seq.size(1):
            raise ValueError(f"Sequence length mismatch in TimeAwareStarGNN: hidden_seq {hidden_seq.size(1)}, time_diffs_seq {time_diffs_seq.size(1)}")

        time_emb = self.time_embed(time_diffs_seq.unsqueeze(-1).float())
        hidden_time_aware_seq = hidden_seq + time_emb

        batch_size, seq_len, _ = hidden_time_aware_seq.size()
        star_nodes = self.star_center.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        augmented_hidden_time_aware_seq = torch.cat([star_nodes, hidden_time_aware_seq], dim=1)

        attn_output, _ = self.attn(
            augmented_hidden_time_aware_seq, augmented_hidden_time_aware_seq, augmented_hidden_time_aware_seq
        )
        attn_output = self.dropout(attn_output)
        star_context = attn_output[:, 0]

        output_seq_items = hidden_time_aware_seq + attn_output[:, 1:]
        output_seq_items = self.norm(output_seq_items)

        output_seq_items = self.ffn(output_seq_items) + output_seq_items
        output_seq_items = self.norm(output_seq_items)

        return output_seq_items, star_context

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

        self.dropout = nn.Dropout(0.3)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                 weight.data.uniform_(-stdv, stdv)
            else:
                 weight.data.uniform_(-stdv, stdv)
        for layer in [self.linear_edge_in, self.linear_edge_out]:
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data.uniform_(-stdv, stdv)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        batch_size, num_nodes_in_hidden, _ = hidden.shape

        expected_A_feature_dim = 2 * num_nodes_in_hidden

        if A.shape[1] != num_nodes_in_hidden:
             raise ValueError(f"Mismatch in A's node dimension ({A.shape[1]}) and hidden's node dimension ({num_nodes_in_hidden})")

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

        self.star_gnn = TimeAwareStarGNN(
            hidden_size=self.hidden_size,
            num_heads=8 # Can be configured via opt if needed
        )

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8, # Can be configured via opt if needed
            dropout=0.3, # Can be configured via opt if needed
            batch_first=True
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True) # For a, ht, star_context
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Unused? Check if this is needed

        self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.2) # Label smoothing can be opt.label_smoothing

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2), # Or make intermediate dim configurable
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 2, projection_dim)
        )
        self.reset_parameters()

        self.dropout = nn.Dropout(0.3) # General dropout, can be opt.dropout

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10, # Can be opt.scheduler_T0
            T_mult=2, # Can be opt.scheduler_Tmult
            eta_min=1e-5 # Can be opt.scheduler_eta_min
        )


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            # Avoid re-initializing sub-modules that have their own reset_parameters
            if name.startswith("tagnn.") or name.startswith("star_gnn."):
                # Check if the base module (tagnn or star_gnn) has reset_parameters
                # This logic is a bit simplified; robust check might involve inspecting the module object itself
                if hasattr(getattr(self, name.split('.')[0]), 'reset_parameters') and name.split('.')[0] in ['tagnn', 'star_gnn']:
                    continue # Skip if sub-module handles its own reset
            
            if 'bias' in name:
                nn.init.zeros_(weight)
            else:
                if 'embedding.weight' in name and 'position_embedding' not in name: # Item embedding
                    nn.init.xavier_uniform_(weight)
                elif 'position_embedding.weight' in name: # Positional embedding
                    nn.init.xavier_uniform_(weight) # Or some other suitable init
                elif weight.dim() > 1: # Other linear layers, etc.
                    weight.data.uniform_(-stdv, stdv)
        
        # Ensure padding_idx in embeddings is zero
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        if self.position_embedding.padding_idx is not None:
            with torch.no_grad():
                self.position_embedding.weight[self.position_embedding.padding_idx].fill_(0)


    def _get_seq_hidden_from_unique_nodes(self, gnn_output_on_unique_nodes, alias_inputs_for_sequence):
        batch_size = gnn_output_on_unique_nodes.size(0)
        # alias_inputs_for_sequence shape: (batch_size, max_seq_len)
        # gnn_output_on_unique_nodes shape: (batch_size, max_unique_nodes_in_batch, hidden_size)
        
        # Expand alias_inputs to gather along the hidden_size dimension
        alias_indices_expanded = alias_inputs_for_sequence.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        seq_item_hidden = torch.gather(
            gnn_output_on_unique_nodes,
            1, # Gather along the unique_nodes dimension
            alias_indices_expanded
        )
        return seq_item_hidden

    def _add_positional_embeddings(self, seq_item_hidden, position_ids_for_sequence):
        # position_ids_for_sequence shape: (batch_size, max_seq_len)
        pos_embeds = self.position_embedding(position_ids_for_sequence.long())
        # pos_embeds shape: (batch_size, max_seq_len, position_emb_dim)

        if self.position_emb_dim == self.hidden_size: # If dims match, just add
            seq_hidden_final = seq_item_hidden + pos_embeds
        else: # If dims don't match, concatenate and project
            combined = torch.cat([seq_item_hidden, pos_embeds], dim=-1)
            seq_hidden_final = self.position_proj(combined)
            
        return seq_hidden_final


    def forward(self, unique_item_inputs, A_matrix, alias_inputs_for_sequence, position_ids_for_sequence, time_diffs_for_sequence, sequence_mask_for_attention):
        # unique_item_inputs: (batch_size, max_unique_nodes_in_batch) - IDs of unique items
        # A_matrix: (batch_size, max_unique_nodes_in_batch, max_unique_nodes_in_batch * 2) - Adjacency matrix
        # alias_inputs_for_sequence: (batch_size, max_seq_len) - Maps seq positions to unique node indices
        # position_ids_for_sequence: (batch_size, max_seq_len) - Positional IDs for each item in sequence
        # time_diffs_for_sequence: (batch_size, max_seq_len) - Time differences for each item
        # sequence_mask_for_attention: (batch_size, max_seq_len) - Bool mask (True for padding) for MHA key_padding_mask

        # 1. Embed unique items
        unique_item_embeds = self.embedding(unique_item_inputs) # (batch_size, max_unique_nodes, hidden_size)
        hidden_gnn_unique_nodes = self.dropout(unique_item_embeds) # Apply dropout

        # 2. TAGNN processing on unique item graph
        hidden_gnn_unique_nodes = self.tagnn(A_matrix, hidden_gnn_unique_nodes) # (batch_size, max_unique_nodes, hidden_size)

        # 3. Map GNN outputs back to sequence format
        seq_hidden_from_gnn = self._get_seq_hidden_from_unique_nodes(hidden_gnn_unique_nodes, alias_inputs_for_sequence) # (batch_size, max_seq_len, hidden_size)

        # 4. Add positional embeddings
        seq_hidden_with_pos = self._add_positional_embeddings(seq_hidden_from_gnn, position_ids_for_sequence) # (batch_size, max_seq_len, hidden_size)

        # 5. Time-Aware Star GNN processing
        # time_diffs_for_sequence needs to be (batch_size, max_seq_len)
        seq_hidden_time_aware, star_context = self.star_gnn(seq_hidden_with_pos, time_diffs_for_sequence)
        # seq_hidden_time_aware: (batch_size, max_seq_len, hidden_size)
        # star_context: (batch_size, hidden_size)

        # 6. Final Self-Attention Layer
        x = seq_hidden_time_aware
        x_norm = self.layer_norm1(x) # Apply LayerNorm before attention
        # sequence_mask_for_attention should be True for positions to be ignored
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, 
                              key_padding_mask=sequence_mask_for_attention) 
        final_seq_hidden = x + self.dropout(x_attn) # Residual connection and dropout

        return final_seq_hidden, star_context


    def compute_scores(self, final_seq_hidden, star_context, mask_for_scoring):
        # final_seq_hidden: (batch_size, max_seq_len, hidden_size)
        # star_context: (batch_size, hidden_size)
        # mask_for_scoring: (batch_size, max_seq_len) - 1 for actual items, 0 for padding

        # Get hidden state of the last actual item (ht)
        actual_lengths = torch.sum(mask_for_scoring, 1).long() # (batch_size)
        # Clamp to avoid negative indices if a sequence is all padding (shouldn't happen with filtering)
        last_item_indices = torch.clamp(actual_lengths - 1, min=0) 
        
        batch_indices = torch.arange(final_seq_hidden.shape[0], 
                                    device=final_seq_hidden.device)
        ht = final_seq_hidden[batch_indices, last_item_indices] # (batch_size, hidden_size)

        # Attention mechanism for session summary 'a'
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1]) # (batch_size, 1, hidden_size)
        q2 = self.linear_two(final_seq_hidden) # (batch_size, max_seq_len, hidden_size)
        
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)) # (batch_size, max_seq_len, 1)
        # Mask out padding before softmax
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -1e9) # Use a large negative number
        alpha = F.softmax(alpha_logits_masked, dim=1) # (batch_size, max_seq_len, 1)

        # Weighted sum of item hidden states
        a = torch.sum(alpha * final_seq_hidden * mask_for_scoring.unsqueeze(-1).float(), dim=1) # (batch_size, hidden_size)

        # Combine features: attentive sum 'a', last item 'ht', and 'star_context'
        combined_features = torch.cat([a, ht, star_context], dim=1) # (batch_size, hidden_size * 3)
        final_session_embedding = self.linear_transform(combined_features) # (batch_size, hidden_size)

        # Compute scores against all candidate items (excluding padding item 0)
        candidate_item_embeddings = self.embedding.weight[1:] # (n_items, hidden_size)
        scores = torch.matmul(final_session_embedding, candidate_item_embeddings.transpose(0, 1)) # (batch_size, n_items)
        return scores

    def get_session_embedding_for_ssl(self, final_seq_hidden, mask_for_ssl):
        # Use the hidden state of the last item in the (potentially augmented) sequence for SSL
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.clamp(actual_lengths - 1, min=0)
        batch_indices = torch.arange(final_seq_hidden.shape[0],
                                    device=final_seq_hidden.device)
        session_repr = final_seq_hidden[batch_indices, last_item_indices] # (batch_size, hidden_size)
        return session_repr

    def calculate_infonce_loss(self, z1, z2):
        # z1, z2: (batch_size, projection_dim)
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        
        sim_matrix = torch.matmul(z1_norm, z2_norm.T) / self.ssl_temperature # (batch_size, batch_size)
        # Positive pairs are on the diagonal
        labels = torch.arange(z1_norm.size(0)).long().to(z1_norm.device)
        
        loss_ssl = F.cross_entropy(sim_matrix, labels)
        return loss_ssl


def train_test(model, train_data, test_data, opt, device): # test_data is now eval_data_loader
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
        # Return eval metrics (0) and training losses (0)
        return 0.0, 0.0, 0.0, 0.0, 0.0 

    for i_slice_indices, j_batch_num in tqdm(zip(slices, np.arange(len(slices))), total=len(slices), desc=f"Epoch {opt.current_epoch_num} Training"):
        if len(i_slice_indices) == 0:
            continue

        model_module.optimizer.zero_grad()

        ssl_drop_prob = opt.ssl_item_drop_prob

        data_v1_tuple, data_v2_tuple, targets_main_np, mask_main_np, time_diffs_v1_np, time_diffs_v2_np = train_data.get_slice(
            i_slice_indices, ssl_item_drop_prob=ssl_drop_prob
        )

        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl, position_ids_v1 = data_v1_tuple
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl, position_ids_v2 = data_v2_tuple


        if items_v1_unique.size == 0 or items_v2_unique.size == 0: # Should be rare due to proc_utils._augment ensuring non-empty
            print(f"Skipping batch {j_batch_num} due to empty unique items array in v1 or v2.")
            continue

        # --- View 1 (Main view for prediction loss, also part of SSL) ---
        items_v1_unique = torch.from_numpy(items_v1_unique).long().to(device)
        A_v1 = torch.from_numpy(A_v1).float().to(device)
        alias_inputs_v1 = torch.from_numpy(alias_inputs_v1).long().to(device)
        mask_v1_ssl = torch.from_numpy(mask_v1_ssl).long().to(device) # This is the mask for the v1 sequence itself
        position_ids_v1 = torch.from_numpy(position_ids_v1).long().to(device)
        time_diffs_v1 = torch.from_numpy(time_diffs_v1_np).float().to(device)

        # --- View 2 (Augmented view for SSL) ---
        items_v2_unique = torch.from_numpy(items_v2_unique).long().to(device)
        A_v2 = torch.from_numpy(A_v2).float().to(device)
        alias_inputs_v2 = torch.from_numpy(alias_inputs_v2).long().to(device)
        mask_v2_ssl = torch.from_numpy(mask_v2_ssl).long().to(device) # This is the mask for the v2 (augmented) sequence
        position_ids_v2 = torch.from_numpy(position_ids_v2).long().to(device)
        time_diffs_v2 = torch.from_numpy(time_diffs_v2_np).float().to(device)

        targets_main = torch.from_numpy(targets_main_np).long().to(device)
        mask_main_cuda = torch.from_numpy(mask_main_np).long().to(device) # Original sequence mask for main prediction
        
        # Attention mask for MHA layer in Attention_SessionGraph (True for padding)
        # For v1, use the original mask (mask_main_cuda or equivalently mask_v1_ssl as v1 is not augmented by item drop)
        seq_attention_mask_v1 = (mask_main_cuda == 0) 

        final_seq_hidden_v1, star_context_v1 = model(items_v1_unique, A_v1, alias_inputs_v1, position_ids_v1, time_diffs_v1, seq_attention_mask_v1)

        if final_seq_hidden_v1.size(0) == 0: # Should not happen if batch has items
            print(f"Skipping batch {j_batch_num} due to empty final_seq_hidden_v1.")
            continue
        
        # Main prediction loss
        scores_main = model_module.compute_scores(final_seq_hidden_v1, star_context_v1, mask_main_cuda) # Use original mask
        loss_main = model_module.loss_function(scores_main, targets_main - 1) # Targets are 1-indexed, scores are 0-indexed relative to candidate items

        # SSL Loss
        session_emb_v1_ssl = model_module.get_session_embedding_for_ssl(final_seq_hidden_v1, mask_v1_ssl) # Use v1's own mask

        # Attention mask for MHA layer for v2
        seq_attention_mask_v2 = (mask_v2_ssl == 0) 
        final_seq_hidden_v2, _ = model(items_v2_unique, A_v2, alias_inputs_v2, position_ids_v2, time_diffs_v2, seq_attention_mask_v2)
        
        if final_seq_hidden_v2.size(0) == 0:
            print(f"Skipping batch {j_batch_num} due to empty final_seq_hidden_v2.")
            # If v2 becomes empty but v1 was not, we might only have main loss.
            # This path should be rare due to _augment_sequence_item_dropout behavior.
            # For simplicity, if this happens, we might skip SSL for this batch or handle it.
            # Current logic: if final_seq_hidden_v2 is empty, SSL loss won't be computed correctly.
            # A robust way is to check if session_emb_v2_ssl is valid before computing SSL loss.
            # However, proc_utils ensures augmented sequences are not empty if original wasn't.
            loss_ssl = torch.tensor(0.0).to(device) # Default if v2 is problematic
        else:
            session_emb_v2_ssl = model_module.get_session_embedding_for_ssl(final_seq_hidden_v2, mask_v2_ssl) # Use v2's own mask

            projected_emb_v1 = model_module.projection_head_ssl(session_emb_v1_ssl)
            projected_emb_v2 = model_module.projection_head_ssl(session_emb_v2_ssl)
            loss_ssl = model_module.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)

        combined_loss = loss_main + model_module.ssl_weight * loss_ssl

        # Gradient clipping (optional, but good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # max_norm can be opt.clip_grad_norm

        combined_loss.backward()
        model_module.optimizer.step()
        model_module.scheduler.step() # Step per batch for CosineAnnealingWarmRestarts

        total_loss_epoch += combined_loss.item()
        total_main_loss_epoch += loss_main.item()
        total_ssl_loss_epoch += loss_ssl.item()
        num_batches_processed += 1

        if num_batches_processed > 0 and j_batch_num > 0 and len(slices) > 5 and j_batch_num % int(len(slices) / 5) == 0:
            print('[%d/%d] Total Loss: %.4f (Main: %.4f, SSL: %.4f)' %
                  (j_batch_num, len(slices), combined_loss.item(), loss_main.item(), loss_ssl.item()))

    # Calculate average losses for the epoch
    avg_total_loss_train = 0.0
    avg_main_loss_train = 0.0
    avg_ssl_loss_train = 0.0
    if num_batches_processed > 0:
        avg_total_loss_train = total_loss_epoch / num_batches_processed
        avg_main_loss_train = total_main_loss_epoch / num_batches_processed
        avg_ssl_loss_train = total_ssl_loss_epoch / num_batches_processed
        print(f'\tTraining Epoch {opt.current_epoch_num} Avg Loss: Total: {avg_total_loss_train:.3f} (Main: {avg_main_loss_train:.3f}, SSL: {avg_ssl_loss_train:.3f})')
    else:
        print(f'\tNo batches were effectively processed in epoch {opt.current_epoch_num}.')

    # --- Evaluation Phase ---
    print(f'Start Prediction/Evaluation for epoch {opt.current_epoch_num}: {datetime.datetime.now()}')
    model.eval()
    hit, mrr = [], []
    # test_data here is the eval_data_loader (either validation or original test set)
    slices_eval = test_data.generate_batch(opt.batchSize) # Use test_data (eval_data_loader)
    
    if not slices_eval:
        print("Warning: No batches generated from evaluation data. Skipping evaluation for this epoch.")
        return 0.0, 0.0, avg_total_loss_train, avg_main_loss_train, avg_ssl_loss_train

    with torch.no_grad():
        for i_eval_slice_indices in tqdm(slices_eval, desc=f"Epoch {opt.current_epoch_num} Evaluation"):
            if len(i_eval_slice_indices) == 0:
                continue
            
            # For evaluation, SSL augmentation is not used (ssl_item_drop_prob=0.0)
            # So v1 is the original sequence, v2 is identical to v1 (or can be ignored)
            data_v1_eval_tuple, _, targets_eval_np, mask_eval_np, time_diffs_eval_np, _ = test_data.get_slice(
                i_eval_slice_indices, ssl_item_drop_prob=0.0 # No dropout for eval
            )

            alias_inputs_eval, A_eval, items_eval_unique, _, position_ids_eval = data_v1_eval_tuple


            if items_eval_unique.size == 0:
                print("Skipping evaluation batch due to empty unique items array.")
                continue

            items_eval_unique = torch.from_numpy(items_eval_unique).long().to(device)
            A_eval = torch.from_numpy(A_eval).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval).long().to(device)
            time_diffs_eval = torch.from_numpy(time_diffs_eval_np).float().to(device)

            mask_eval_cuda = torch.from_numpy(mask_eval_np).long().to(device)
            targets_eval_cuda = torch.from_numpy(targets_eval_np).long().to(device)
            
            seq_attention_mask_eval = (mask_eval_cuda == 0) # True for padding

            final_seq_hidden_eval, star_context_eval = model(items_eval_unique, A_eval, alias_inputs_eval, position_ids_eval, time_diffs_eval, seq_attention_mask_eval)

            if final_seq_hidden_eval.size(0) == 0:
                print("Skipping evaluation batch due to empty final_seq_hidden_eval.")
                continue

            scores_eval = model_module.compute_scores(final_seq_hidden_eval, star_context_eval, mask_eval_cuda)

            # Get top K predictions
            sub_scores_top20_indices = scores_eval.topk(20)[1] # Get indices of top 20 items
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            
            targets_eval_for_metric_np = targets_eval_cuda.cpu().detach().numpy() - 1 # Adjust target to be 0-indexed for comparison with prediction indices

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_for_metric_np):
                hit.append(np.isin(target_item, score_row))
                if target_item in score_row:
                    rank = np.where(score_row == target_item)[0][0] + 1 # Rank is 1-based
                    mrr.append(1.0 / rank)
                else:
                    mrr.append(0.0)

    eval_hit_metric = np.mean(hit) * 100 if hit else 0.0
    eval_mrr_metric = np.mean(mrr) * 100 if mrr else 0.0
    
    return eval_hit_metric, eval_mrr_metric, avg_total_loss_train, avg_main_loss_train, avg_ssl_loss_train