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
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        # A: [batch, max_unique_nodes, 2 * max_unique_nodes]
        # hidden: [batch, max_unique_nodes, hidden_size]
        # A.shape[1] should be max_unique_nodes
        
        # Check if A has the expected 3rd dimension for splitting
        if A.shape[2] == 2 * A.shape[1]:
             input_in_adj = A[:, :, :A.shape[1]]
             input_out_adj = A[:, :, A.shape[1]: 2 * A.shape[1]]
        else: # Fallback or error if A is not as expected (e.g. already split or different format)
            # This might happen if A was already processed differently. Assuming original format for now.
            # For robustness, one might want to check A's construction in Dataset.
            # If A is [batch, max_unique, max_unique] for in and out separately, logic needs change.
            # Based on original `np.concatenate([u_A_in, u_A_out]).transpose()`
            # it means A is [batch, N, 2N] where N is max_unique_nodes for that batch construction.
            # So A.shape[1] is correct for N.
            input_in_adj = A[:, :, :hidden.shape[1]] # Use hidden.shape[1] as num_nodes more reliably if A is padded
            input_out_adj = A[:, :, hidden.shape[1]:]


        input_in = torch.matmul(input_in_adj, self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(input_out_adj, self.linear_edge_out(hidden)) + self.b_oah

        inputs = torch.cat([input_in, input_out], 2) # [batch, max_unique_nodes, 2 * hidden_size]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh) # hidden is [batch, max_unique_nodes, hidden_size]
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
        self.n_node = n_node # Total number of unique items in dataset + 1 for padding
        # self.batch_size removed as it's dynamic
        self.nonhybrid = opt.nonhybrid
        # padding_idx=0 assumes item ID 0 is reserved for padding throughout the system
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0)
        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        # batch_first=True means input/output shape is (N, L, E)
        # N=batch_size, L=sequence_length, E=embedding_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=2, dropout=0.1, batch_first=True)

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.ssl_weight = opt.ssl_weight
        self.ssl_temperature = opt.ssl_temperature
        projection_dim = opt.ssl_projection_dim # This should be calculated in train.py and passed via opt
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, projection_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def _get_seq_hidden_from_gnn_output(self, gnn_output_on_unique_nodes, alias_inputs_for_sequence):
        batch_size = gnn_output_on_unique_nodes.size(0)
        seq_hidden_list = []
        for b_idx in range(batch_size):
            unique_node_embeddings = gnn_output_on_unique_nodes[b_idx]
            alias_indices = alias_inputs_for_sequence[b_idx]
            # Ensure alias_indices are long for indexing
            seq_embeds = unique_node_embeddings[alias_indices.long()]
            seq_hidden_list.append(seq_embeds)
        return torch.stack(seq_hidden_list)

    def forward(self, unique_item_inputs, A_matrix):
        # unique_item_inputs: [batch, max_unique_nodes_in_batch] (item IDs)
        # A_matrix: [batch, max_unique_nodes_in_batch, 2 * max_unique_nodes_in_batch]
        hidden = self.embedding(unique_item_inputs) # [batch, max_unique_nodes, hidden_size]
        hidden = self.tagnn(A_matrix, hidden)       # [batch, max_unique_nodes, hidden_size]

        # Transformer key_padding_mask: True for positions to be masked (padded unique items)
        # unique_item_inputs are IDs, 0 is padding ID.
        transformer_key_padding_mask = (unique_item_inputs == self.embedding.padding_idx) # [N, S_unique]

        hidden_norm = self.layer_norm1(hidden) # Apply layernorm before residual
        # Q, K, V are the same for self-attention
        hidden_attn, _ = self.attn(hidden_norm, hidden_norm, hidden_norm,
                                   key_padding_mask=transformer_key_padding_mask)
        hidden = hidden + hidden_attn # Residual connection (original code had skip = norm(hidden), then attn(hidden) + skip)
                                      # Common practice is norm(x + attn(x)) or x + attn(norm(x)). Let's use x + attn(norm(x))
        # Corrected residual: hidden = hidden_norm + hidden_attn
        # Let's stick to the original structure if layer_norm1 was meant for pre-attention normalization:
        # skip = self.layer_norm1(hidden)
        # hidden, attn_w = self.attn(skip, skip, skip, key_padding_mask=transformer_key_padding_mask)
        # hidden = hidden + skip # This is from original code before SSL, but applied to GNN output, not unique_item embeds.
        # The previous version had:
        # skip = self.layer_norm1(hidden) # hidden is output of GNN here.
        # hidden, attn_w = self.attn(hidden, hidden, hidden, key_padding_mask=transformer_attn_mask) # This was original error, QKV should be from `skip` or `hidden_norm`
        # hidden = hidden+skip
        # Corrected Transformer block application:
        # x = hidden # from GNN
        # x_norm = self.layer_norm1(x)
        # x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=transformer_key_padding_mask)
        # hidden = x + x_attn # Common Post-LN
        # Or Pre-LN:
        x = hidden
        x_norm = self.layer_norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=transformer_key_padding_mask)
        hidden = x + x_attn # Using Post-LN structure as it's simpler. Original skip logic might be different.

        return hidden # [batch, max_unique_nodes, hidden_size]

    def compute_scores(self, seq_hidden, mask_for_scoring):
        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        last_item_indices = torch.max(torch.zeros_like(actual_lengths), actual_lengths - 1)
        ht = seq_hidden[torch.arange(seq_hidden.shape[0]).long(), last_item_indices]

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(seq_hidden)
        alpha_unnormalized = self.linear_three(torch.sigmoid(q1 + q2))
        alpha_unnormalized = alpha_unnormalized.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -1e9) # mask before softmax
        alpha = F.softmax(alpha_unnormalized, 1)
        a = torch.sum(alpha * seq_hidden * mask_for_scoring.unsqueeze(-1).float(), 1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        b = self.embedding.weight[1:] # Candidate item embeddings (all items except padding idx 0)
        scores = torch.matmul(a, b.transpose(0, 1))
        return scores

    def get_session_embedding_for_ssl(self, seq_hidden, mask_for_ssl):
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.max(torch.zeros_like(actual_lengths), actual_lengths - 1)
        session_repr = seq_hidden[torch.arange(seq_hidden.shape[0]).long(), last_item_indices]
        return session_repr

    def calculate_infonce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim_matrix = torch.matmul(z1, z2.T) / self.ssl_temperature
        labels = torch.arange(z1.size(0)).long().to(z1.device) # Ensure labels are on same device
        loss_ssl = F.cross_entropy(sim_matrix, labels)
        return loss_ssl

# Removed to_cuda and to_cpu, will handle device transfer in train_test directly for clarity with DataParallel
# def to_cuda(input_variable): ...
# def to_cpu(input_variable): ...


# train_test now accepts device
def train_test(model, train_data, test_data, opt, device): # <--- device added
    if opt.epoch > 0 :
         # Access original model's scheduler if wrapped by DataParallel
        scheduler_to_step = model.module.scheduler if isinstance(model, torch.nn.DataParallel) else model.scheduler
        scheduler_to_step.step()

    print('Start training: ', datetime.datetime.now())
    model.train() # Set model to training mode

    total_loss_epoch = 0.0
    total_main_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0

    current_batch_size = opt.batchSize
    slices = train_data.generate_batch(current_batch_size)

    for i_slice_indices, j_batch_num in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        # Access original model's optimizer if wrapped by DataParallel
        optimizer_to_use = model.module.optimizer if isinstance(model, torch.nn.DataParallel) else model.optimizer
        optimizer_to_use.zero_grad()

        ssl_drop_prob = opt.ssl_item_drop_prob
        data_v1, data_v2, targets_main, mask_main = train_data.get_slice(i_slice_indices, ssl_item_drop_prob=ssl_drop_prob)

        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl = data_v1
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl = data_v2

        # Move data to the target device (e.g., "cuda:0" for DataParallel, or "cpu")
        items_v1_unique = torch.Tensor(items_v1_unique).long().to(device)
        A_v1 = torch.Tensor(A_v1).float().to(device)
        alias_inputs_v1 = torch.Tensor(alias_inputs_v1).long().to(device) # Ensure long for indexing
        mask_v1_ssl = torch.Tensor(mask_v1_ssl).long().to(device)

        items_v2_unique = torch.Tensor(items_v2_unique).long().to(device)
        A_v2 = torch.Tensor(A_v2).float().to(device)
        alias_inputs_v2 = torch.Tensor(alias_inputs_v2).long().to(device) # Ensure long for indexing
        mask_v2_ssl = torch.Tensor(mask_v2_ssl).long().to(device)

        targets_main = torch.Tensor(targets_main).long().to(device)
        mask_main = torch.Tensor(mask_main).long().to(device)

        # --- Main Task (using view 1) ---
        # Model forward pass for unique items of view 1
        # `model` here is potentially the DataParallel wrapper
        gnn_output_v1 = model(items_v1_unique, A_v1) # [batch_actual, max_unique_v1, hidden_size]

        # Reconstruct full sequence hidden states for view 1
        # If using DataParallel, access underlying model methods via .module
        _get_seq_hidden_func = model.module._get_seq_hidden_from_gnn_output if isinstance(model, torch.nn.DataParallel) else model._get_seq_hidden_from_gnn_output
        seq_hidden_v1 = _get_seq_hidden_func(gnn_output_v1, alias_inputs_v1)

        # Compute recommendation scores
        _compute_scores_func = model.module.compute_scores if isinstance(model, torch.nn.DataParallel) else model.compute_scores
        scores_main = _compute_scores_func(seq_hidden_v1, mask_main)
        
        _loss_func = model.module.loss_function if isinstance(model, torch.nn.DataParallel) else model.loss_function
        loss_main = _loss_func(scores_main, targets_main - 1)

        # --- SSL Task (Contrasting v1 and v2) ---
        _get_session_emb_ssl_func = model.module.get_session_embedding_for_ssl if isinstance(model, torch.nn.DataParallel) else model.get_session_embedding_for_ssl
        session_emb_v1_ssl = _get_session_emb_ssl_func(seq_hidden_v1, mask_v1_ssl)

        gnn_output_v2 = model(items_v2_unique, A_v2)
        seq_hidden_v2 = _get_seq_hidden_func(gnn_output_v2, alias_inputs_v2)
        session_emb_v2_ssl = _get_session_emb_ssl_func(seq_hidden_v2, mask_v2_ssl)

        _projection_head_ssl_func = model.module.projection_head_ssl if isinstance(model, torch.nn.DataParallel) else model.projection_head_ssl
        projected_emb_v1 = _projection_head_ssl_func(session_emb_v1_ssl)
        projected_emb_v2 = _projection_head_ssl_func(session_emb_v2_ssl)

        _calc_infonce_loss_func = model.module.calculate_infonce_loss if isinstance(model, torch.nn.DataParallel) else model.calculate_infonce_loss
        loss_ssl = _calc_infonce_loss_func(projected_emb_v1, projected_emb_v2)

        # Total loss
        ssl_weight_val = model.module.ssl_weight if isinstance(model, torch.nn.DataParallel) else model.ssl_weight
        combined_loss = loss_main + ssl_weight_val * loss_ssl

        combined_loss.backward()
        optimizer_to_use.step()

        total_loss_epoch += combined_loss.item()
        total_main_loss_epoch += loss_main.item()
        total_ssl_loss_epoch += loss_ssl.item()

        if j_batch_num % int(len(slices) / 5 + 1) == 0 and len(slices) > 0:
            print('[%d/%d] Total Loss: %.4f (Main: %.4f, SSL: %.4f)' %
                  (j_batch_num, len(slices), combined_loss.item(), loss_main.item(), loss_ssl.item()))

    if len(slices) > 0:
        avg_total_loss = total_loss_epoch / len(slices)
        avg_main_loss = total_main_loss_epoch / len(slices)
        avg_ssl_loss = total_ssl_loss_epoch / len(slices)
        print('\tTraining Epoch Loss (Avg): Total: %.3f (Main: %.3f, SSL: %.3f)' % (avg_total_loss, avg_main_loss, avg_ssl_loss))
    else:
        print('\tNo training batches were processed.')

    # Evaluation
    print('Start Prediction: ', datetime.datetime.now())
    model.eval() # Set model to evaluation mode
    hit, mrr = [], []
    slices_test = test_data.generate_batch(opt.batchSize)

    with torch.no_grad():
        for i_test_slice_indices in slices_test:
            data_v1_test, _, targets_test_orig, mask_test_orig = test_data.get_slice(i_test_slice_indices, ssl_item_drop_prob=0.0) # No SSL aug for eval
            alias_inputs_eval, A_eval, items_eval_unique, _ = data_v1_test

            items_eval_unique = torch.Tensor(items_eval_unique).long().to(device)
            A_eval = torch.Tensor(A_eval).float().to(device)
            alias_inputs_eval = torch.Tensor(alias_inputs_eval).long().to(device)
            mask_test_orig_cuda = torch.Tensor(mask_test_orig).long().to(device)
            targets_test_orig_cuda = torch.Tensor(targets_test_orig).long().to(device)

            gnn_output_eval = model(items_eval_unique, A_eval) # Model is potentially DataParallel

            _get_seq_hidden_func_eval = model.module._get_seq_hidden_from_gnn_output if isinstance(model, torch.nn.DataParallel) else model._get_seq_hidden_from_gnn_output
            seq_hidden_eval = _get_seq_hidden_func_eval(gnn_output_eval, alias_inputs_eval)

            _compute_scores_func_eval = model.module.compute_scores if isinstance(model, torch.nn.DataParallel) else model.compute_scores
            scores_eval = _compute_scores_func_eval(seq_hidden_eval, mask_test_orig_cuda)

            sub_scores_top20_indices = scores_eval.topk(20)[1]
            sub_scores_top20_indices_np = sub_scores_top20_indices.cpu().detach().numpy() # Move to CPU for numpy
            targets_eval_np = targets_test_orig_cuda.cpu().detach().numpy() # Move to CPU for numpy

            for score_row, target_item in zip(sub_scores_top20_indices_np, targets_eval_np):
                target_for_eval = target_item - 1
                hit.append(np.isin(target_for_eval, score_row))
                if target_for_eval in score_row:
                    mrr.append(1 / (np.where(score_row == target_for_eval)[0][0] + 1))
                else:
                    mrr.append(0)

    hit_metric = np.mean(hit) * 100 if hit else 0
    mrr_metric = np.mean(mrr) * 100 if mrr else 0
    return hit_metric, mrr_metric

# get_mask and get_pos seem unused, can be kept or removed if confirmed.
def get_mask(seq_len):
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))

def get_pos(seq_len):
    return torch.arange(seq_len).unsqueeze(0)
