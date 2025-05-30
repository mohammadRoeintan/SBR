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
    def __init__(self, hidden_size, step=1):
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

    def GNNCell(self, A, hidden):
        # A: [batch, max_unique_nodes, 2 * max_unique_nodes]
        # hidden: [batch, max_unique_nodes, hidden_size]
        num_nodes_in_hidden = hidden.shape[1]

        # Ensure A's relevant dimension matches hidden's node dimension
        if A.shape[1] != num_nodes_in_hidden or A.shape[2] != 2 * num_nodes_in_hidden:
            # This might occur if batch_max_n_node used in A construction differs from hidden.shape[1]
            # Or if A is not correctly shaped as N x 2N.
            # A robust GNN cell might need to handle A parts separately or ensure consistency.
            # For now, assume A is correctly formed based on the number of nodes present in `hidden`.
            # This means A should be [batch, num_nodes_in_hidden, 2 * num_nodes_in_hidden]
            # A simple fix if A is larger due to batch padding: slice A
            if A.shape[1] > num_nodes_in_hidden:
                A = A[:, :num_nodes_in_hidden, :2*num_nodes_in_hidden]
            # If A is still not matching, this indicates a deeper issue in graph construction / batching
            if A.shape[2] != 2 * num_nodes_in_hidden:
                 print(f"Warning: GNNCell A shape {A.shape} mismatch with hidden shape {hidden.shape}. Graph processing might be incorrect.")
                 # Fallback, assuming A is [N, N] for in and [N, N] for out, concatenated.
                 # This part needs to be certain based on how A is constructed.
                 # The current A construction aims for A_in and A_out based on max_n_node_batch, then concatenated.
                 # So hidden.shape[1] should indeed be this max_n_node_batch.
                 input_in_adj = A[:, :, :num_nodes_in_hidden]
                 input_out_adj = A[:, :, num_nodes_in_hidden:2*num_nodes_in_hidden] # if A is N x 2N
            else:
                 input_in_adj = A[:, :, :num_nodes_in_hidden]
                 input_out_adj = A[:, :, num_nodes_in_hidden:2*num_nodes_in_hidden]

        input_in = torch.matmul(input_in_adj, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(input_out_adj, self.linear_edge_out(hidden)) + self.b_oah

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
            print(f"Warning: position_emb_dim ({self.position_emb_dim}) is not equal to hidden_size ({self.hidden_size}). "
                  "Ensure combination logic (e.g., projection or careful concatenation) is correctly handled if not using addition.")
            # For addition, they should be equal. If using concatenation, this is fine.
            # The current _get_seq_hidden_with_position uses addition.
            # Forcing them to be equal for addition if not specified.
            # self.position_emb_dim = self.hidden_size # Or handle projection
            
        self.position_embedding = nn.Embedding(self.max_len + 1, self.position_emb_dim, padding_idx=0)

        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=2, dropout=0.1, batch_first=True)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.loss_function = nn.CrossEntropyLoss()

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, projection_dim)
        )
        self.reset_parameters()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(weight)
            else:
                if 'embedding.weight' in name: # Item embedding
                    nn.init.xavier_uniform_(weight)
                elif 'position_embedding.weight' in name: # Position embedding
                    nn.init.xavier_uniform_(weight)
                else:
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
        
        # Ensure alias_inputs are long and correctly shaped for gather/indexing
        alias_indices_expanded = alias_inputs_for_sequence.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        # gnn_output_on_unique_nodes is [batch, max_unique_nodes, hidden_size]
        # We need to gather along dim 1 (max_unique_nodes)
        seq_item_hidden = torch.gather(gnn_output_on_unique_nodes, 1, alias_indices_expanded)
        # Result: [batch, max_seq_len, hidden_size]
        
        pos_embeds = self.position_embedding(position_ids_for_sequence.long())
        
        if self.position_emb_dim == self.hidden_size:
            seq_hidden_final = seq_item_hidden + pos_embeds
        else:
            # Fallback: if dims don't match for addition, simply use item embeddings
            # Or implement a projection layer for pos_embeds if this is intended.
            print(f"Critical Warning: position_emb_dim ({self.position_emb_dim}) != hidden_size ({self.hidden_size}). "
                  "Positional embeddings not added. Fix dimensions or projection.")
            seq_hidden_final = seq_item_hidden
            
        return seq_hidden_final


    def forward(self, unique_item_inputs, A_matrix):
        hidden = self.embedding(unique_item_inputs)
        hidden = self.tagnn(A_matrix, hidden)
        
        transformer_key_padding_mask = (unique_item_inputs == self.embedding.padding_idx)
        
        x = hidden
        x_norm = self.layer_norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=transformer_key_padding_mask)
        hidden = x + x_attn # Post-LN structure
        
        return hidden


    def compute_scores(self, seq_hidden_time_aware, mask_for_scoring):
        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        # Ensure indices are valid even for all-padding sequences (though mask should prevent this affecting loss)
        last_item_indices = torch.max(torch.zeros_like(actual_lengths, device=actual_lengths.device), actual_lengths - 1)
        
        ht = seq_hidden_time_aware[torch.arange(seq_hidden_time_aware.shape[0]).long(), last_item_indices]

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_hidden_time_aware)
        alpha_unnormalized = self.linear_three(torch.sigmoid(q1 + q2))
        alpha_unnormalized = alpha_unnormalized.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -torch.finfo(alpha_unnormalized.dtype).max) # More stable masking
        alpha = F.softmax(alpha_unnormalized, 1)
        a = torch.sum(alpha * seq_hidden_time_aware * mask_for_scoring.unsqueeze(-1).float(), 1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        # Assuming item ID 0 is padding and not a valid target item
        candidate_item_embeddings = self.embedding.weight[1:]
        scores = torch.matmul(a, candidate_item_embeddings.transpose(0, 1))
        return scores

    def get_session_embedding_for_ssl(self, seq_hidden_time_aware, mask_for_ssl):
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.max(torch.zeros_like(actual_lengths, device=actual_lengths.device), actual_lengths - 1)
        session_repr = seq_hidden_time_aware[torch.arange(seq_hidden_time_aware.shape[0]).long(), last_item_indices]
        return session_repr

    def calculate_infonce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim_matrix = torch.matmul(z1, z2.T) / self.ssl_temperature
        labels = torch.arange(z1.size(0)).long().to(z1.device)
        loss_ssl = F.cross_entropy(sim_matrix, labels)
        return loss_ssl


def train_test(model, train_data, test_data, opt, device):
    # Access attributes from the original model if wrapped by DataParallel
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    if opt.epoch > 0 and model_module.scheduler.last_epoch < opt.epoch :
        model_module.scheduler.step()

    print('Start training: ', datetime.datetime.now())
    model.train()

    total_loss_epoch = 0.0
    total_main_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0

    current_batch_size = opt.batchSize
    slices = train_data.generate_batch(current_batch_size)
    if not slices:
        print("Warning: No batches generated from training data. Skipping training epoch.")
        return 0.0, 0.0 # Return zero metrics if no training happened

    for i_slice_indices, j_batch_num in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        if len(i_slice_indices) == 0: continue # Skip if a slice is empty

        model_module.optimizer.zero_grad()

        ssl_drop_prob = opt.ssl_item_drop_prob
        data_v1, data_v2, targets_main_np, mask_main_np = train_data.get_slice(i_slice_indices, ssl_item_drop_prob=ssl_drop_prob)

        # Skip batch if data is empty (can happen if get_slice returns empty arrays for an empty slice)
        if data_v1[0].size == 0 or data_v2[0].size == 0:
            print(f"Skipping empty batch {j_batch_num}")
            continue

        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl, position_ids_v1 = data_v1
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl, position_ids_v2 = data_v2

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

        gnn_output_v1 = model(items_v1_unique, A_v1)
        seq_hidden_v1_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v1, alias_inputs_v1, position_ids_v1)
        scores_main = model_module.compute_scores(seq_hidden_v1_time_aware, mask_main)
        loss_main = model_module.loss_function(scores_main, targets_main - 1)

        session_emb_v1_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v1_time_aware, mask_v1_ssl)
        
        gnn_output_v2 = model(items_v2_unique, A_v2)
        seq_hidden_v2_time_aware = model_module._get_seq_hidden_with_position(gnn_output_v2, alias_inputs_v2, position_ids_v2)
        session_emb_v2_ssl = model_module.get_session_embedding_for_ssl(seq_hidden_v2_time_aware, mask_v2_ssl)

        projected_emb_v1 = model_module.projection_head_ssl(session_emb_v1_ssl)
        projected_emb_v2 = model_module.projection_head_ssl(session_emb_v2_ssl)
        loss_ssl = model_module.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)

        combined_loss = loss_main + model_module.ssl_weight * loss_ssl
        combined_loss.backward()
        model_module.optimizer.step()

        total_loss_epoch += combined_loss.item(); total_main_loss_epoch += loss_main.item(); total_ssl_loss_epoch += loss_ssl.item()

        if j_batch_num > 0 and len(slices) > 5 and j_batch_num % int(len(slices) / 5) == 0 : # Print more frequently
            print('[%d/%d] Total Loss: %.4f (Main: %.4f, SSL: %.4f)' %
                  (j_batch_num, len(slices), combined_loss.item(), loss_main.item(), loss_ssl.item()))
    
    if len(slices) > 0 and total_loss_epoch > 0: # Check if any training happened
        avg_total_loss = total_loss_epoch / len(slices)
        avg_main_loss = total_main_loss_epoch / len(slices)
        avg_ssl_loss = total_ssl_loss_epoch / len(slices)
        print('\tTraining Epoch Loss (Avg): Total: %.3f (Main: %.3f, SSL: %.3f)' % (avg_total_loss, avg_main_loss, avg_ssl_loss))
    else:
        print('\tNo training batches were effectively processed or loss was zero.')

    print('Start Prediction: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices_test = test_data.generate_batch(opt.batchSize)
    if not slices_test:
        print("Warning: No batches generated from test data. Skipping evaluation.")
        return 0.0, 0.0

    with torch.no_grad():
        for i_test_slice_indices in slices_test:
            if len(i_test_slice_indices) == 0: continue

            data_v1_test, _, targets_test_orig_np, mask_test_orig_np = test_data.get_slice(i_test_slice_indices, ssl_item_drop_prob=0.0)
            
            if data_v1_test[0].size == 0: # Skip if slice resulted in empty data
                print(f"Skipping empty test batch.")
                continue

            alias_inputs_eval, A_eval, items_eval_unique, _, position_ids_eval = data_v1_test

            items_eval_unique = torch.from_numpy(items_eval_unique).long().to(device)
            A_eval = torch.from_numpy(A_eval).float().to(device)
            alias_inputs_eval = torch.from_numpy(alias_inputs_eval).long().to(device)
            position_ids_eval = torch.from_numpy(position_ids_eval).long().to(device)
            mask_test_orig_cuda = torch.from_numpy(mask_test_orig_np).long().to(device)
            targets_test_orig_cuda = torch.from_numpy(targets_test_orig_np).long().to(device)

            gnn_output_eval = model(items_eval_unique, A_eval)
            seq_hidden_eval_time_aware = model_module._get_seq_hidden_with_position(gnn_output_eval, alias_inputs_eval, position_ids_eval)
            scores_eval = model_module.compute_scores(seq_hidden_eval_time_aware, mask_test_orig_cuda)

            sub_scores_top20_indices = scores_eval.topk(20)[1]
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy()
            targets_eval_np = targets_test_orig_cuda.cpu().detach().numpy()

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_np):
                target_for_eval = target_item - 1
                hit.append(np.isin(target_for_eval, score_row))
                if target_for_eval in score_row:
                    mrr.append(1 / (np.where(score_row == target_for_eval)[0][0] + 1))
                else:
                    mrr.append(0)

    hit_metric = np.mean(hit) * 100 if hit else 0.0
    mrr_metric = np.mean(mrr) * 100 if mrr else 0.0
    return hit_metric, mrr_metric
