############################################################
# This code builds on https://github.com/CRIPAC-DIG/TAGNN #
# and integrates STAR: https://github.com/yeganegi-reza/STAR #
############################################################
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
    def __init__(self, hidden_size, time_embed_dim=8, num_heads=8):
        super().__init__()
        self.time_embed = nn.Linear(time_embed_dim, hidden_size)
        self.star_center = nn.Parameter(torch.Tensor(hidden_size))
        
        # مقداردهی اولیه مناسب برای تانسور 1 بعدی
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

    def forward(self, hidden, time_diffs):
        # Time-aware embedding
        time_emb = self.time_embed(time_diffs.unsqueeze(-1))
        hidden = hidden + time_emb
        
        # Star topology integration
        batch_size, seq_len, _ = hidden.size()
        star_nodes = self.star_center.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)
        augmented_hidden = torch.cat([star_nodes, hidden], dim=1)
        
        # Star-attention mechanism
        attn_output, _ = self.attn(
            augmented_hidden, augmented_hidden, augmented_hidden
        )
        attn_output = self.dropout(attn_output)
        star_context = attn_output[:, 0]  # Extract star node representation
        
        # Residual connection and FFN
        output = augmented_hidden + attn_output
        output = self.norm(output)
        
        # Feedforward network
        output = self.ffn(output) + output
        output = self.norm(output)
        
        return output[:, 1:], star_context

# بقیه کد بدون تغییر (همانند قبل)...

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
        
        # STAR integration
        self.star_gnn = TimeAwareStarGNN(
            hidden_size=self.hidden_size,
            time_embed_dim=8,
            num_heads=8
        )
        
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=0.2)

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size * 2, projection_dim)
        )
        self.reset_parameters()
        
        self.dropout = nn.Dropout(0.3)
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-5
        )


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name.startswith("tagnn.") or name.startswith("star_gnn."): 
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
        max_seq_len_in_batch = alias_inputs_for_sequence.size(1)
        
        alias_indices_expanded = alias_inputs_for_sequence.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        seq_item_hidden = torch.gather(
            gnn_output_on_unique_nodes, 
            1, 
            alias_indices_expanded
        )
            
        pos_embeds = self.position_embedding(position_ids_for_sequence.long())
        
        if self.position_emb_dim == self.hidden_size:
            seq_hidden_final = seq_item_hidden + pos_embeds
        else:
            combined = torch.cat([seq_item_hidden, pos_embeds], dim=-1)
            seq_hidden_final = self.position_proj(combined)
            
        return seq_hidden_final


    def forward(self, unique_item_inputs, A_matrix, time_diffs):
        hidden = self.embedding(unique_item_inputs)
        hidden = self.dropout(hidden)
        hidden = self.tagnn(A_matrix, hidden)
        
        # STAR integration
        hidden, star_context = self.star_gnn(hidden, time_diffs)
        
        transformer_key_padding_mask = (unique_item_inputs == self.embedding.padding_idx)
        
        x = hidden
        x_norm = self.layer_norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, 
                              key_padding_mask=transformer_key_padding_mask)
        hidden = x + self.dropout(x_attn)

        return hidden, star_context


    def compute_scores(self, seq_hidden_time_aware, star_context, mask_for_scoring):
        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        last_item_indices = torch.clamp(actual_lengths - 1, min=0)
        
        batch_indices = torch.arange(seq_hidden_time_aware.shape[0], 
                                    device=seq_hidden_time_aware.device)
        ht = seq_hidden_time_aware[batch_indices, last_item_indices]

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_hidden_time_aware)
        
        alpha_logits = self.linear_three(torch.sigmoid(q1 + q2))
        alpha_logits_masked = alpha_logits.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(alpha_logits_masked, dim=1)
        
        a = torch.sum(alpha * seq_hidden_time_aware * mask_for_scoring.unsqueeze(-1).float(), dim=1)
        
        # Combine with STAR context
        combined = torch.cat([a, ht, star_context], dim=1)
        a = self.linear_transform(combined)

        candidate_item_embeddings = self.embedding.weight[1:] 
        scores = torch.matmul(a, candidate_item_embeddings.transpose(0, 1))
        return scores

    def get_session_embedding_for_ssl(self, seq_hidden_time_aware, mask_for_ssl):
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.clamp(actual_lengths - 1, min=0)
        batch_indices = torch.arange(seq_hidden_time_aware.shape[0], 
                                    device=seq_hidden_time_aware.device)
        session_repr = seq_hidden_time_aware[batch_indices, last_item_indices]
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
        data_v1, data_v2, targets_main_np, mask_main_np, time_diffs_v1, time_diffs_v2 = train_data.get_slice(i_slice_indices, ssl_item_drop_prob=ssl_drop_prob)

        if data_v1[0].size == 0 or data_v2[0].size == 0:
            continue

        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl, position_ids_v1 = data_v1
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl, position_ids_v2 = data_v2

        items_v1_unique = torch.from_numpy(items_v1_unique).long().to(device)
        A_v1 = torch.from_numpy(A_v1).float().to(device)
        alias_inputs_v1 = torch.from_numpy(alias_inputs_v1).long().to(device)
        mask_v1_ssl = torch.from_numpy(mask_v1_ssl).long().to(device)
        position_ids_v1 = torch.from_numpy(position_ids_v1).long().to(device)
        time_diffs_v1 = torch.from_numpy(time_diffs_v1).float().to(device)

        items_v2_unique = torch.from_numpy(items_v2_unique).long().to(device)
        A_v2 = torch.from_numpy(A_v2).float().to(device)
        alias_inputs_v2 = torch.from_numpy(alias_inputs_v2).long().to(device)
        mask_v2_ssl = torch.from_numpy(mask_v2_ssl).long().to(device)
        position_ids_v2 = torch.from_numpy(position_ids_v2).long().to(device)
        time_diffs_v2 = torch.from_numpy(time_diffs_v2).float().to(device)

        targets_main = torch.from_numpy(targets_main_np).long().to(device)
        mask_main = torch.from_numpy(mask_main_np).long().to(device)

        gnn_output_v1, star_context_v1 = model(items_v1_unique, A_v1, time_diffs_v1)
        if gnn_output_v1.size(0) == 0:
            continue
            
        seq_hidden_v1_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v1, alias_inputs_v1, position_ids_v1)
        scores_main = model_module.compute_scores(seq_hidden_v1_time_aware, star_context_v1, mask_main)
        loss_main = model_module.loss_function(scores_main, targets_main - 1)

        session_emb_v1_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v1_time_aware, mask_v1_ssl)
        
        gnn_output_v2, star_context_v2 = model(items_v2_unique, A_v2, time_diffs_v2)
        if gnn_output_v2.size(0) == 0:
            continue
        seq_hidden_v2_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v2, alias_inputs_v2, position_ids_v2)
        session_emb_v2_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v2_time_aware, mask_v2_ssl)

        projected_emb_v1 = model_module.projection_head_ssl(session_emb_v1_ssl)
        projected_emb_v2 = model_module.projection_head_ssl(session_emb_v2_ssl)
        loss_ssl = model_module.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)

        combined_loss = loss_main + model_module.ssl_weight * loss_ssl
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        combined_loss.backward()
        model_module.optimizer.step()
        model_module.scheduler.step()

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

            data_v1_test, _, targets_test_orig_np, mask_test_orig_np, time_diffs_test, _ = test_data.get_slice(i_test_slice_indices, ssl_item_drop_prob=0.0)
            
            if data_v1_test[0].size == 0: 
                continue

            alias_inputs_eval, A_eval, items_eval_unique, _, position_ids_eval = data_v1_test

            items_eval_unique = torch.from_numpy(items_eval_unique).long().to(device)
            A_eval = torch.from_numpy(A_eval).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval).long().to(device)
            time_diffs_test = torch.from_numpy(time_diffs_test).float().to(device)
            mask_test_orig_cuda = torch.from_numpy(mask_test_orig_np).long().to(device)
            targets_test_orig_cuda = torch.from_numpy(targets_test_orig_np).long().to(device)

            gnn_output_eval, star_context_eval = model(items_eval_unique, A_eval, time_diffs_test)
            if gnn_output_eval.size(0) == 0: 
                continue

            seq_hidden_eval_time_aware = model_module._get_seq_hidden_with_position(gnn_output_eval, alias_inputs_eval, position_ids_eval)
            if seq_hidden_eval_time_aware.size(0) == 0: 
                continue
                
            scores_eval = model_module.compute_scores(seq_hidden_eval_time_aware, star_context_eval, mask_test_orig_cuda)

            sub_scores_top20_indices = scores_eval.topk(20)[1]
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
