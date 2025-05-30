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

# from agc import AGC # Already removed


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
        input_in = torch.matmul(A[:, :, :A.shape[1]], # Corrected slicing for A
                                self.linear_edge_in(hidden)) + self.b_iah

        input_out = torch.matmul(
            A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah # Corrected slicing for A

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
        self.batch_size = opt.batchSize # Note: batch_size can vary for the last batch
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0) # Assuming 0 is padding
        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention( # Transformer Encoder part
            embed_dim=self.hidden_size, num_heads=2, dropout=0.1, batch_first=True) # Added batch_first=True

        self.linear_one = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear( # target attention for candidate items
            self.hidden_size, self.hidden_size, bias=False)
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        # SSL specific parameters
        self.ssl_weight = opt.ssl_weight if hasattr(opt, 'ssl_weight') else 0.1
        self.ssl_temperature = opt.ssl_temperature if hasattr(opt, 'ssl_temperature') else 0.07
        projection_dim = opt.ssl_projection_dim if hasattr(opt, 'ssl_projection_dim') else self.hidden_size // 2
        self.projection_head_ssl = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), # First layer of projection
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, projection_dim) # Second layer to final projection dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # Ensure padding_idx in embedding is zeroed out if used and not trained
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)


    def _get_seq_hidden_from_gnn_output(self, gnn_output_on_unique_nodes, alias_inputs_for_sequence):
        # gnn_output_on_unique_nodes: [batch, max_unique_nodes_in_batch, hidden_size]
        # alias_inputs_for_sequence: [batch, max_seq_len_in_batch] (indices into unique_nodes)
        batch_size = gnn_output_on_unique_nodes.size(0)
        seq_hidden_list = []
        for b_idx in range(batch_size):
            unique_node_embeddings = gnn_output_on_unique_nodes[b_idx] # [max_unique_nodes, hidden_size]
            alias_indices = alias_inputs_for_sequence[b_idx]          # [max_seq_len]
            
            # Gather embeddings based on alias_indices
            # Ensure alias_indices are within bounds of unique_node_embeddings.size(0)
            # Valid alias indices should be < unique_node_embeddings.size(0)
            # Padding in alias_indices (usually 0) should map to padding embedding (usually also at index 0 of unique_node_embeddings if 0 is a unique node)
            seq_embeds = unique_node_embeddings[alias_indices]
            seq_hidden_list.append(seq_embeds)
        
        return torch.stack(seq_hidden_list) # [batch, max_seq_len, hidden_size]

    # This is the main forward pass for the GNN + Transformer
    def forward(self, unique_item_inputs, A_matrix):
        # unique_item_inputs: [batch, max_unique_nodes_in_batch], contains IDs of unique items in sessions
        # A_matrix: [batch, max_unique_nodes_in_batch, 2 * max_unique_nodes_in_batch], adjacency matrices
        
        hidden = self.embedding(unique_item_inputs) # [batch, max_unique_nodes, hidden_size]
        hidden = self.tagnn(A_matrix, hidden)       # [batch, max_unique_nodes, hidden_size] (after GNN)
        
        # The original code had permute operations for a non-batch_first Transformer.
        # If using nn.MultiheadAttention with batch_first=True, input should be (N, L, E)
        # N=batch_size, L=sequence_length (here, max_unique_nodes), E=embedding_dim (hidden_size)
        
        # Transformer Encoder part expects (L, N, E) if batch_first=False (default)
        # or (N, L, E) if batch_first=True.
        # Output of tagnn is (N, L, E) = (batch, max_unique_nodes, hidden_size) which is suitable for batch_first=True.

        # Create attention mask for transformer: True means position is masked.
        # Mask where unique_item_inputs are padding (e.g. ID 0).
        # This mask is for the sequence of *unique items*, not the full session sequence.
        # (N, S) where S is source sequence length.
        transformer_attn_mask = (unique_item_inputs == self.embedding.padding_idx) # [batch, max_unique_nodes]
        
        skip = self.layer_norm1(hidden) # Apply layernorm before residual
        # For self-attention, query, key, value are the same.
        # src_key_padding_mask should be FloatTensor with -inf for masked positions and 0.0 for unmasked.
        # Or BoolTensor where True indicates masking.
        # MultiheadAttention expects key_padding_mask of shape (N, S)
        hidden, attn_w = self.attn(hidden, hidden, hidden, key_padding_mask=transformer_attn_mask)
        hidden = hidden + skip # Residual connection
        
        # Output 'hidden' is now the processed embeddings for the unique items in each session
        # Shape: [batch, max_unique_nodes, hidden_size]
        return hidden

    def compute_scores(self, seq_hidden, mask_for_scoring):
        # seq_hidden: [batch, max_session_len, hidden_size], full sequential hidden states
        # mask_for_scoring: [batch, max_session_len], mask for the actual items in session (1 for item, 0 for padding)
        
        # Get last hidden state ht based on the mask
        # Sum mask to get sequence lengths, subtract 1 for 0-based index
        actual_lengths = torch.sum(mask_for_scoring, 1).long()
        # Ensure lengths are at least 1 to avoid negative indices, if a sequence is all padding, ht will be based on index 0.
        last_item_indices = torch.max(torch.zeros_like(actual_lengths), actual_lengths - 1)
        
        ht = seq_hidden[torch.arange(seq_hidden.shape[0]).long(), last_item_indices]  # [batch_size, hidden_size]
        
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # [batch_size, 1, hidden_size]
        q2 = self.linear_two(seq_hidden)  # [batch_size, max_session_len, hidden_size]
        
        alpha_unnormalized = self.linear_three(torch.sigmoid(q1 + q2)) # [batch_size, max_session_len, 1]
        # Apply mask before softmax: set alpha for padded positions to a very small number
        alpha_unnormalized = alpha_unnormalized.masked_fill(mask_for_scoring.unsqueeze(-1) == 0, -1e9)
        alpha = F.softmax(alpha_unnormalized, 1)  # [batch_size, max_session_len, 1] (softmax over seq_len dim)
        
        # Weighted sum of hidden states
        # mask.view(...).float() ensures only actual items contribute
        a = torch.sum(alpha * seq_hidden * mask_for_scoring.unsqueeze(-1).float(), 1) # [batch_size, hidden_size]

        if not self.nonhybrid: # If hybrid, combine with last item state
            a = self.linear_transform(torch.cat([a, ht], 1))
            
        b = self.embedding.weight[1:]  # Candidate item embeddings (all items except padding idx 0)
                                       # Shape: [n_node-1, hidden_size]

        # Scores for all candidate items
        # Dot product between session embedding 'a' and all item embeddings 'b'
        # 'a': [batch_size, hidden_size]
        # 'b': [n_node-1, hidden_size]
        # Result: [batch_size, n_node-1]
        # This part was more complex in original, involving target attention. Let's simplify first.
        # Original target attention:
        # hidden_masked = seq_hidden * mask_for_scoring.view(mask_for_scoring.shape[0], -1, 1).float()
        # qt = self.linear_t(hidden_masked)  # [batch_size, max_session_len, hidden_size]
        # beta = F.softmax(b @ qt.transpose(1, 2), -1) # [batch_size, n_node-1, max_session_len]
        # target_aware_session_emb = beta @ hidden_masked # [batch_size, n_node-1, hidden_size]
        # a_expanded = a.view(a.shape[0], 1, a.shape[1])  # [batch_size, 1, hidden_size]
        # combined_emb_for_scoring = a_expanded + target_aware_session_emb # [batch_size, n_node-1, hidden_size]
        # scores = torch.sum(combined_emb_for_scoring * b.unsqueeze(0), -1) # [batch_size, n_node-1]
        # Simpler scoring:
        scores = torch.matmul(a, b.transpose(0, 1)) # [batch_size, n_node-1]

        return scores

    def get_session_embedding_for_ssl(self, seq_hidden, mask_for_ssl):
        # seq_hidden: [batch, max_session_len, hidden_size] (for an augmented view)
        # mask_for_ssl: [batch, max_session_len] (mask for this view)
        # For SSL, use the representation of the last valid item in the sequence.
        actual_lengths = torch.sum(mask_for_ssl, 1).long()
        last_item_indices = torch.max(torch.zeros_like(actual_lengths), actual_lengths - 1)
        
        session_repr = seq_hidden[torch.arange(seq_hidden.shape[0]).long(), last_item_indices]
        return session_repr

    def calculate_infonce_loss(self, z1, z2): 
        # z1, z2: [batch_size, projection_dim], projected embeddings of two views
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Cosine similarity matrix
        #sim_matrix = torch.matmul(z1, z2.T) / self.ssl_temperature # [batch_size, batch_size]
        # یک طرف را detach می‌کنیم تا از collapse projection جلوگیری کنیم
        sim_matrix = torch.matmul(z1, z2.detach().T) / self.ssl_temperature


        # Positive pairs are on the diagonal (i-th session in view1 corresponds to i-th session in view2)
        labels = torch.arange(z1.size(0)).long().to(z1.device)
        
        loss_ssl = F.cross_entropy(sim_matrix, labels)
        return loss_ssl


def get_mask(seq_len): # This seems to be for a causal mask in Transformer if not using key_padding_mask
    # For self-attention over unique items, key_padding_mask is more direct.
    # If this is for the Attention_GNN's GNNCell, it's not used there.
    # If for the MultiheadAttention, it might be an alternative, but key_padding_mask is standard.
    # Let's assume it's not critical for now given the current structure.
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool'))#.to('cuda') # Device handled by caller


def to_cuda(input_variable):
    if torch.cuda.is_available():
        return input_variable.cuda()
    else:
        return input_variable


def to_cpu(input_variable):
    if torch.cuda.is_available():
        return input_variable.cpu()
    else:
        return input_variable


# Standalone forward function for evaluation (modified)
def forward_eval(model, i_indices_batch, data_loader, opt):
    # For eval, SSL augmentation is turned off (drop_prob=0)
    # data_v1 will be the original sequence data.
    # targets_main and mask_main will be for the original sequence.
    data_v1, _, targets_main, mask_main = data_loader.get_slice(i_indices_batch, ssl_item_drop_prob=0.0)
    
    alias_inputs_eval, A_eval, items_eval_unique_nodes, _ = data_v1 # mask_v1_ssl not used for scoring

    items_eval_unique_nodes = to_cuda(torch.Tensor(items_eval_unique_nodes).long())
    A_eval = to_cuda(torch.Tensor(A_eval).float())
    alias_inputs_eval = to_cuda(torch.Tensor(alias_inputs_eval).long())
    mask_main_cuda = to_cuda(torch.Tensor(mask_main).long()) # Original mask for scoring
    targets_main_cuda = to_cuda(torch.Tensor(targets_main).long())

    # 1. Pass unique items and their graph through GNN+Transformer part of the model
    gnn_output_on_unique_nodes = model(items_eval_unique_nodes, A_eval) 
    # Output: [batch, max_unique_nodes_in_batch, hidden_size]

    # 2. Reconstruct full sequential hidden states using alias_inputs
    seq_hidden_eval = model._get_seq_hidden_from_gnn_output(gnn_output_on_unique_nodes, alias_inputs_eval)
    # Output: [batch, max_session_len, hidden_size]
           
    # 3. Compute recommendation scores
    scores = model.compute_scores(seq_hidden_eval, mask_main_cuda)
    
    return targets_main_cuda, scores


def train_test(model, train_data, test_data, opt): # Added opt
    if opt.epoch > 0 : # Only step scheduler if actually training
        model.scheduler.step() # Step LR scheduler each epoch
        
    print('Start training: ', datetime.datetime.now())
    model.train()
    total_loss_epoch = 0.0
    total_main_loss_epoch = 0.0
    total_ssl_loss_epoch = 0.0
    
    # Fetch batch_size from opt, as model.batch_size might not be updated for last batch
    current_batch_size = opt.batchSize 
    slices = train_data.generate_batch(current_batch_size)

    for i_slice_indices, j_batch_num in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        model.optimizer.zero_grad()
        
        # Get data for two views + original targets and mask for main task
        ssl_drop_prob = opt.ssl_item_drop_prob if hasattr(opt, 'ssl_item_drop_prob') else 0.2
        data_v1, data_v2, targets_main, mask_main = train_data.get_slice(i_slice_indices, ssl_item_drop_prob=ssl_drop_prob)

        alias_inputs_v1, A_v1, items_v1_unique, mask_v1_ssl = data_v1
        alias_inputs_v2, A_v2, items_v2_unique, mask_v2_ssl = data_v2

        # Move data to CUDA
        items_v1_unique = to_cuda(torch.Tensor(items_v1_unique).long())
        A_v1 = to_cuda(torch.Tensor(A_v1).float())
        alias_inputs_v1 = to_cuda(torch.Tensor(alias_inputs_v1).long())
        mask_v1_ssl = to_cuda(torch.Tensor(mask_v1_ssl).long())

        items_v2_unique = to_cuda(torch.Tensor(items_v2_unique).long())
        A_v2 = to_cuda(torch.Tensor(A_v2).float())
        alias_inputs_v2 = to_cuda(torch.Tensor(alias_inputs_v2).long())
        mask_v2_ssl = to_cuda(torch.Tensor(mask_v2_ssl).long())
        
        targets_main = to_cuda(torch.Tensor(targets_main).long())
        mask_main = to_cuda(torch.Tensor(mask_main).long()) # Original mask for main task scoring

        # --- Main Task (using view 1, which is original or very mildly augmented) ---
        # 1. GNN+Transformer forward pass for unique items of view 1
        gnn_output_v1 = model(items_v1_unique, A_v1) # [batch, max_unique_v1, hidden_size]
        # 2. Reconstruct full sequence hidden states for view 1
        seq_hidden_v1 = model._get_seq_hidden_from_gnn_output(gnn_output_v1, alias_inputs_v1) # [batch, max_seq_len, hidden_size]
        # 3. Compute recommendation scores using original mask
        scores_main = model.compute_scores(seq_hidden_v1, mask_main) 
        loss_main = model.loss_function(scores_main, targets_main - 1) # Ensure targets are 0-indexed if necessary for loss

        # --- SSL Task (Contrasting v1 and v2) ---
        # `seq_hidden_v1` is already computed.
        # For SSL, we need a session-level embedding from `seq_hidden_v1` using `mask_v1_ssl`.
        session_emb_v1_ssl = model.get_session_embedding_for_ssl(seq_hidden_v1, mask_v1_ssl)
        
        # Compute for view 2
        gnn_output_v2 = model(items_v2_unique, A_v2)
        seq_hidden_v2 = model._get_seq_hidden_from_gnn_output(gnn_output_v2, alias_inputs_v2)
        session_emb_v2_ssl = model.get_session_embedding_for_ssl(seq_hidden_v2, mask_v2_ssl)

        # Project session embeddings for SSL
        projected_emb_v1 = model.projection_head_ssl(session_emb_v1_ssl)
        projected_emb_v2 = model.projection_head_ssl(session_emb_v2_ssl)
           
        loss_ssl = model.calculate_infonce_loss(projected_emb_v1, projected_emb_v2)
           
        # Total loss
        #combined_loss = loss_main + model.ssl_weight * loss_ssl
        # warm-up: در ۲ epoch اول ssl_weight = 0.0
        current_ssl_weight = 0.0 if opt.epoch <= 2 else model.ssl_weight
        combined_loss = loss_main + current_ssl_weight * loss_ssl  

        combined_loss.backward()
        model.optimizer.step()
           
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
    model.eval()
    hit, mrr = [], []
    # Use opt.batchSize for test data loader as well
    slices_test = test_data.generate_batch(opt.batchSize)

    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        for i_test_slice_indices in slices_test:
            # The forward_eval function handles getting data with ssl_item_drop_prob=0.0
            targets_eval, scores_eval = forward_eval(model, i_test_slice_indices, test_data, opt)
            
            sub_scores_top20_indices = scores_eval.topk(20)[1] # Get indices of top 20 items
            sub_scores_top20_indices = to_cpu(sub_scores_top20_indices).detach().numpy()
            
            targets_eval_np = to_cpu(targets_eval).detach().numpy()
            
            # Get the original mask for the test batch to know valid lengths, if needed by metric calc,
            # but typically hit/mrr are calculated per session.
            # The `test_data.mask` might not be aligned if generate_batch shuffles.
            # It's safer if `get_slice` also returns the original mask for test data for this purpose if needed.
            # However, the current loop iterates over `targets_eval_np` which is already per session.
            # The `mask` in `zip` in original code for test was `test_data.mask` - this assumes test_data isn't shuffled
            # or that `mask` is fetched in sync. The current loop structure is fine.

            for score_row, target_item in zip(sub_scores_top20_indices, targets_eval_np):
                # score_row are predicted item indices (0 to N-2, as target is target-1)
                # target_item is original ID, so target_item-1 is the 0-indexed version
                target_for_eval = target_item - 1 
                hit.append(np.isin(target_for_eval, score_row))
                if target_for_eval in score_row:
                    mrr.append(1 / (np.where(score_row == target_for_eval)[0][0] + 1))
                else:
                    mrr.append(0)

    hit_metric = np.mean(hit) * 100 if hit else 0
    mrr_metric = np.mean(mrr) * 100 if mrr else 0
    return hit_metric, mrr_metric


def get_pos(seq_len): # Seems unused currently
    return torch.arange(seq_len).unsqueeze(0)
