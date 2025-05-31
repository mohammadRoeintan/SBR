############################################################
# This code builds on https://github.com/CRIPAC-DIG/TAGNN #
# and integrates STAR: https://github.com/yeganegi-reza/STAR #
############################################################

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
        # نگاشت تفاوت زمانی (اسکالر) به فضای پنهان
        self.time_embed = nn.Linear(1, hidden_size)
        
        # پارامتر گره ستاره مرکزی
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

    def forward(self, hidden_seq, time_diffs_seq): # hidden_seq should be (batch, seq_len, hidden_size)
        # جاسازی زمانی: تبدیل شکل time_diffs به (batch_size, seq_len, 1)
        # Ensure time_diffs_seq corresponds to hidden_seq in seq_len
        if hidden_seq.size(1) != time_diffs_seq.size(1):
            # This check is crucial, but the root cause should be fixed in data prep if it triggers
            raise ValueError(f"Sequence length mismatch in TimeAwareStarGNN: hidden_seq {hidden_seq.size(1)}, time_diffs_seq {time_diffs_seq.size(1)}")
        
        time_emb = self.time_embed(time_diffs_seq.unsqueeze(-1).float())
        hidden_time_aware_seq = hidden_seq + time_emb # This was the error line
        
        # ادغام توپولوژی ستاره
        batch_size, seq_len, _ = hidden_time_aware_seq.size()
        star_nodes = self.star_center.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        # Prepend star_nodes to each sequence
        augmented_hidden_time_aware_seq = torch.cat([star_nodes, hidden_time_aware_seq], dim=1)
        
        # مکانیزم توجه ستاره
        attn_output, _ = self.attn(
            augmented_hidden_time_aware_seq, augmented_hidden_time_aware_seq, augmented_hidden_time_aware_seq
        )
        attn_output = self.dropout(attn_output)
        star_context = attn_output[:, 0]  # استخراج نمایندگی گره ستاره (representation of star node)
        
        # اتصال باقیمانده و FFN for sequence items only
        output_seq_items = hidden_time_aware_seq + attn_output[:, 1:] # Apply to original sequence items
        output_seq_items = self.norm(output_seq_items)
        
        # شبکه فیدفوروارد
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
        
        # Dropout
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
        # A: (batch, num_unique_nodes, 2 * num_unique_nodes)
        # hidden: (batch, num_unique_nodes, hidden_size)
        batch_size, num_nodes_in_hidden, _ = hidden.shape
        
        # The adjacency matrix A is defined over unique nodes.
        # A_compat needs to match dimensions based on num_nodes_in_hidden.
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
        
        self.max_len = opt.max_len # This should be the sequence max_len
        self.position_emb_dim = opt.position_emb_dim
        if self.position_emb_dim != self.hidden_size:
            self.position_proj = nn.Linear(self.hidden_size + self.position_emb_dim, self.hidden_size)
            self.requires_projection = True
        else:
            self.requires_projection = False
            self.position_proj = None
            
        self.position_embedding = nn.Embedding(self.max_len + 1, self.position_emb_dim, padding_idx=0) # max_len for sequences

        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)
        
        self.star_gnn = TimeAwareStarGNN(
            hidden_size=self.hidden_size,
            num_heads=8 # Assuming 8 heads, can be configurable
        )
        
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8, # Assuming 8 heads
            dropout=0.3, # Assuming 0.3 dropout
            batch_first=True
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True) # For combining a, ht, star_context
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Unused?
        
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.2) # Assuming 0.2 label smoothing

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 2, projection_dim)
        )
        self.reset_parameters()
        
        self.dropout = nn.Dropout(0.3) # General dropout
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, # Example T_0
            T_mult=2, # Example T_mult
            eta_min=1e-5 # Example eta_min
        )


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name.startswith("tagnn.") or name.startswith("star_gnn."): 
                # Let sub-modules initialize themselves if they have their own reset_parameters
                if hasattr(getattr(self, name.split('.')[0]), 'reset_parameters') and name.split('.')[0] in ['tagnn', 'star_gnn']:
                    continue # Skip if sub-module has its own reset
            if 'bias' in name:
                nn.init.zeros_(weight)
            else:
                if 'embedding.weight' in name and 'position_embedding' not in name:
                    nn.init.xavier_uniform_(weight)
                elif 'position_embedding.weight' in name:
                    nn.init.xavier_uniform_(weight)
                elif weight.dim() > 1: # For other linear layers etc.
                    weight.data.uniform_(-stdv, stdv)
        
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)
        if self.position_embedding.padding_idx is not None:
            with torch.no_grad():
                self.position_embedding.weight[self.position_embedding.padding_idx].fill_(0)


    def _get_seq_hidden_from_unique_nodes(self, gnn_output_on_unique_nodes, alias_inputs_for_sequence):
        # gnn_output_on_unique_nodes: (batch, num_unique_nodes, hidden_size)
        # alias_inputs_for_sequence: (batch, seq_len) - maps seq item to its index in unique_nodes
        
        batch_size = gnn_output_on_unique_nodes.size(0)
        # seq_len obtained from alias_inputs_for_sequence
        
        alias_indices_expanded = alias_inputs_for_sequence.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        # Gather operation to map unique node embeddings back to sequence structure
        seq_item_hidden = torch.gather(
            gnn_output_on_unique_nodes, 
            1, # dim to gather along (the unique_nodes dim)
            alias_indices_expanded # (batch, seq_len, hidden_size)
        )
        return seq_item_hidden

    def _add_positional_embeddings(self, seq_item_hidden, position_ids_for_sequence):
        # seq_item_hidden: (batch, seq_len, hidden_size)
        # position_ids_for_sequence: (batch, seq_len)
        pos_embeds = self.position_embedding(position_ids_for_sequence.long()) # (batch, seq_len, pos_emb_dim)
        
        if self.position_emb_dim == self.hidden_size:
            seq_hidden_final = seq_item_hidden + pos_embeds
        else:
            combined = torch.cat([seq_item_hidden, pos_embeds], dim=-1)
            seq_hidden_final = self.position_proj(combined)
            
        return seq_hidden_final


    def forward(self, unique_item_inputs, A_matrix, alias_inputs_for_sequence, position_ids_for_sequence, time_diffs_for_sequence, sequence_mask_for_attention):
        # unique_item_inputs: (batch, num_unique_nodes) - Padded unique items in batch
        # A_matrix: (batch, num_unique_nodes, 2 * num_unique_nodes) - Adjacency for unique items
        # alias_inputs_for_sequence: (batch, seq_len) - Maps seq items to unique item indices
        # position_ids_for_sequence: (batch, seq_len) - Positional IDs for seq items
        # time_diffs_for_sequence: (batch, seq_len) - Time diffs for each seq item
        # sequence_mask_for_attention: (batch, seq_len) - Boolean mask (True for padding)

        # 1. Get initial embeddings for unique items
        unique_item_embeds = self.embedding(unique_item_inputs) # (batch, num_unique_nodes, hidden_size)
        hidden_gnn_unique_nodes = self.dropout(unique_item_embeds)
        
        # 2. Pass unique item embeddings through GNN (TAGNN)
        # This operates on the graph of unique items
        hidden_gnn_unique_nodes = self.tagnn(A_matrix, hidden_gnn_unique_nodes) # (batch, num_unique_nodes, hidden_size)

        # 3. Map GNN outputs from unique nodes back to sequence structure
        seq_hidden_from_gnn = self._get_seq_hidden_from_unique_nodes(hidden_gnn_unique_nodes, alias_inputs_for_sequence)
                                                                # (batch, seq_len, hidden_size)
        
        # 4. Add positional embeddings to the sequence item embeddings
        seq_hidden_with_pos = self._add_positional_embeddings(seq_hidden_from_gnn, position_ids_for_sequence)
                                                            # (batch, seq_len, hidden_size)

        # 5. Apply TimeAwareStarGNN to the sequence-structured, position-aware embeddings
        # time_diffs_for_sequence should have the same seq_len as seq_hidden_with_pos
        seq_hidden_time_aware, star_context = self.star_gnn(seq_hidden_with_pos, time_diffs_for_sequence)
                                                        # (batch, seq_len, hidden_size), (batch, hidden_size)
        
        # 6. Further processing (e.g., another attention layer over the sequence)
        # The original code had a transformer_key_padding_mask based on unique_item_inputs,
        # which is incorrect here as we are operating on sequences.
        # sequence_mask_for_attention should be used.
        x = seq_hidden_time_aware
        x_norm = self.layer_norm1(x) # Layer norm on sequence embeddings
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, 
                              key_padding_mask=sequence_mask_for_attention) # (batch, seq_len, hidden_size)
        final_seq_hidden = x + self.dropout(x_attn) # Residual connection

        return final_seq_hidden, star_context


    def compute_scores(self, final_seq_hidden, star_context, mask_for_scoring):
        # final_seq_hidden: (batch, seq_len, hidden_size) - Output from forward()
        # star_context: (batch, hidden_size) - Star context from star_gnn
        # mask_for_scoring: (batch, seq_len) - Indicates valid items in sequence (1 for valid, 0 for padding)

        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        last_item_indices = torch.clamp(actual_lengths - 1, min=0) # Ensure indices are valid
        
        batch_indices = torch.arange(final_seq_hidden.shape[0], 
                                    device=final_seq_hidden.device)
        ht = final_seq_hidden[batch_indices, last_item_indices] # (batch, hidden_size) - Last item's hidden state

        # Attention mechanism to get session embedding 'a'
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1]) # (batch, 1, hidden_size)
        q2 = self.linear_two(final_seq_hidden) # (batch, seq_len, hidden_size)
        
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2)) # (batch, seq_len, 1)
        # Apply mask before softmax: fill padding positions with a very small number
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(alpha_logits_masked, dim=1) # (batch, seq_len, 1) - Attention weights
        
        # Weighted sum of item embeddings based on attention weights
        # mask_for_scoring is used to ensure only valid items contribute
        a = torch.sum(alpha * final_seq_hidden * mask_for_scoring.unsqueeze(-1).float(), dim=1) # (batch, hidden_size)
        
        # Combine session embedding 'a', last item hidden state 'ht', and star_context
        combined_features = torch.cat([a, ht, star_context], dim=1) # (batch, hidden_size * 3)
        final_session_embedding = self.linear_transform(combined_features) # (batch, hidden_size)

        # Compute scores against all candidate items (excluding padding item 0)
        candidate_item_embeddings = self.embedding.weight[1:] # (n_node-1, hidden_size)
        scores = torch.matmul(final_session_embedding, candidate_item_embeddings.transpose(0, 1)) # (batch, n_node-1)
        return scores

    def get_session_embedding_for_ssl(self, final_seq_hidden, mask_for_ssl):
        # final_seq_hidden: (batch, seq_len, hidden_size)
        # mask_for_ssl: (batch, seq_len) - Mask for the SSL view
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.clamp(actual_lengths - 1, min=0)
        batch_indices = torch.arange(final_seq_hidden.shape[0], 
                                    device=final_seq_hidden.device)
        session_repr = final_seq_hidden[batch_indices, last_item_indices] # Use last item's repr for SSL
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
        # get_slice now needs to return alias_inputs, A, items_unique, mask_ssl, positions, time_diffs, AND sequence_mask_for_attention
        # Let's assume get_slice is updated to provide these correctly.
        # data_v1 = (alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl, position_ids_v1, time_diffs_v1, seq_attention_mask_v1)
        # data_v2 = (alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl, position_ids_v2, time_diffs_v2, seq_attention_mask_v2)

        data_v1, data_v2, targets_main_np, mask_main_np, _, _ = train_data.get_slice(
            i_slice_indices, ssl_item_drop_prob=ssl_drop_prob
        ) # Original get_slice returns 6 items, the last two are time_diffs for v1 and v2
        
        # Unpack data from get_slice carefully
        # (alias_inputs, A_matrix, unique_item_ids, mask_for_view_items, position_ids, time_diffs_for_view_items)
        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl, position_ids_v1, time_diffs_v1 = data_v1
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl, position_ids_v2, time_diffs_v2 = data_v2


        if items_v1_unique.size == 0 or items_v2_unique.size == 0: # Check based on unique items array
            print("Skipping batch due to empty unique items array.")
            continue

        # Convert to tensors
        items_v1_unique = torch.from_numpy(items_v1_unique).long().to(device)
        A_v1 = torch.from_numpy(A_v1).float().to(device)
        alias_inputs_v1 = torch.from_numpy(alias_inputs_v1).long().to(device) # seq_len
        mask_v1_ssl = torch.from_numpy(mask_v1_ssl).long().to(device) # seq_len, used for SSL repr
        position_ids_v1 = torch.from_numpy(position_ids_v1).long().to(device) # seq_len
        time_diffs_v1 = torch.from_numpy(time_diffs_v1).float().to(device) # seq_len

        items_v2_unique = torch.from_numpy(items_v2_unique).long().to(device)
        A_v2 = torch.from_numpy(A_v2).float().to(device)
        alias_inputs_v2 = torch.from_numpy(alias_inputs_v2).long().to(device)
        mask_v2_ssl = torch.from_numpy(mask_v2_ssl).long().to(device)
        position_ids_v2 = torch.from_numpy(position_ids_v2).long().to(device)
        time_diffs_v2 = torch.from_numpy(time_diffs_v2).float().to(device)

        targets_main = torch.from_numpy(targets_main_np).long().to(device)
        # mask_main_np is (batch, seq_len), 1 for item, 0 for pad. This is good for compute_scores and attention key_padding_mask
        mask_main_cuda = torch.from_numpy(mask_main_np).long().to(device)
        # For MultiheadAttention, key_padding_mask should be True for padded elements
        seq_attention_mask_v1 = (mask_main_cuda == 0) # True for pads

        # --- View 1 (Main prediction and part of SSL) ---
        # The forward pass now expects more arguments
        final_seq_hidden_v1, star_context_v1 = model(items_v1_unique, A_v1, alias_inputs_v1, position_ids_v1, time_diffs_v1, seq_attention_mask_v1)
        
        if final_seq_hidden_v1.size(0) == 0: # Should check batch dimension
            print("Skipping batch due to empty final_seq_hidden_v1.")
            continue
            
        scores_main = model_module.compute_scores(final_seq_hidden_v1, star_context_v1, mask_main_cuda)
        loss_main = model_module.loss_function(scores_main, targets_main - 1) # targets are 1-indexed

        session_emb_v1_ssl = model_module.get_session_embedding_for_ssl(final_seq_hidden_v1, mask_v1_ssl) # mask_v1_ssl is item mask for view 1
        
        # --- View 2 (Other part of SSL) ---
        # mask_v2_ssl is the item mask for view 2. Need attention mask for view 2.
        # For SSL, the structure of view 2 is important. Assume mask_v2_ssl indicates valid items in its sequence.
        seq_attention_mask_v2 = (mask_v2_ssl == 0) # True for pads in view 2's sequence structure

        final_seq_hidden_v2, star_context_v2 = model(items_v2_unique, A_v2, alias_inputs_v2, position_ids_v2, time_diffs_v2, seq_attention_mask_v2)
        
        if final_seq_hidden_v2.size(0) == 0:
            print("Skipping batch due to empty final_seq_hidden_v2.")
            continue
        session_emb_v2_ssl = model_module.get_session_embedding_for_ssl(final_seq_hidden_v2, mask_v2_ssl)

        projected_emb_v1 = model_module.projection_head_ssl(session_emb_v1_ssl)
        projected_emb_v2 = model_module.projection_head_ssl(session_emb_v2_ssl)
        loss_ssl = model_module.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)

        combined_loss = loss_main + model_module.ssl_weight * loss_ssl
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        
        combined_loss.backward()
        model_module.optimizer.step()
        model_module.scheduler.step() # Step the scheduler

        total_loss_epoch += combined_loss.item()
        total_main_loss_epoch += loss_main.item()
        total_ssl_loss_epoch += loss_ssl.item()
        num_batches_processed += 1

        if num_batches_processed > 0 and j_batch_num > 0 and len(slices) > 5 and j_batch_num % int(len(slices) / 5) == 0:
            print('[%d/%d] Total Loss: %.4f (Main: %.4f, SSL: %.4f)' %
                  (j_batch_num, len(slices), combined_loss.item(), loss_main.item(), loss_ssl.item()))
    
    if num_batches_processed > 0:
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
            
            # For evaluation, ssl_item_drop_prob is 0.0. View 2 is not used.
            data_v1_test, _, targets_test_orig_np, mask_test_orig_np, _, _ = test_data.get_slice(
                i_test_slice_indices, ssl_item_drop_prob=0.0
            )
            
            alias_inputs_eval, A_eval, items_eval_unique, _, position_ids_eval, time_diffs_eval = data_v1_test

            if items_eval_unique.size == 0: 
                print("Skipping test batch due to empty unique items array.")
                continue

            items_eval_unique = torch.from_numpy(items_eval_unique).long().to(device)
            A_eval = torch.from_numpy(A_eval).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval).long().to(device)
            time_diffs_eval = torch.from_numpy(time_diffs_eval).float().to(device) # seq_len
            
            mask_test_orig_cuda = torch.from_numpy(mask_test_orig_np).long().to(device) # (batch, seq_len) item mask
            targets_test_orig_cuda = torch.from_numpy(targets_test_orig_np).long().to(device)
            seq_attention_mask_eval = (mask_test_orig_cuda == 0) # True for pads

            final_seq_hidden_eval, star_context_eval = model(items_eval_unique, A_eval, alias_inputs_eval, position_ids_eval, time_diffs_eval, seq_attention_mask_eval)
            
            if final_seq_hidden_eval.size(0) == 0: 
                print("Skipping test batch due to empty final_seq_hidden_eval.")
                continue
                
            scores_eval = model_module.compute_scores(final_seq_hidden_eval, star_context_eval, mask_test_orig_cuda)

            sub_scores_top20_indices = scores_eval.topk(20)[1] # Get top 20 indices
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            targets_eval_np = targets_test_orig_cuda.cpu().detach().numpy() - 1 # Adjust for 0-indexed scores

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_np):
                hit.append(np.isin(target_item, score_row))
                if target_item in score_row:
                    rank = np.where(score_row == target_item)[0][0] + 1
                    mrr.append(1.0 / rank)
                else:
                    mrr.append(0.0)

    hit_metric = np.mean(hit) * 100 if hit else 0.0
    mrr_metric = np.mean(mrr) * 100 if mrr else 0.0
    return hit_metric, mrr_metric