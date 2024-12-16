import numpy as np
import torch
import torch.nn as nn



class MoETransformer(nn.Module):
    
    def __init__(self, seq_len, num_patch, num_layer, d_in, d_out, d_patch, d_model, d_hidden, d_qkv, num_head, num_expert, dropout, top_k, training_flag, mask_flag, mask_rate, importance_factor, load_factor, device):
        super(MoETransformer, self).__init__()

        self.d_model = d_model
        self.training_flag = training_flag
        self.mask_flag = mask_flag
        self.mask_rate = mask_rate
        self.num_encoder_layer = int(num_layer/2)
        self.encoder = nn.ModuleList([MoEEncoderLayer(2*d_model, d_hidden, d_qkv, num_head, num_expert, dropout, top_k, training_flag, importance_factor, load_factor, device) for _ in range(self.num_encoder_layer)])
        self.patch_layer = DataPatching(seq_len, d_in, d_patch)
        self.embedding_layer = EmbeddingPE(d_patch, d_model*2, dropout, device)
        self.output_layer = DecoderOutput(seq_len, num_patch, d_patch, d_model, d_out, dropout)
        self.softplus = nn.Softplus()
        self.device = device

    def forward(self, x):
        # input --- x: seq_len * batch * d_in
        x = self.patch_layer(x)             # x: patch_num * batch * patch_dim
        if self.training_flag or self.mask_flag:
            x = self.patch_masking(x, self.mask_rate)    # x: patch_num * batch * patch_dim
        mask_data = x * (self.patch_layer.data_norm_std + 1e-5) + self.patch_layer.data_norm_mean
        x = self.embedding_layer(x)         # x: patch_num * batch * (2*d_model)

        aux_loss_list = []
        for encoder_layer in self.encoder:
            x, aux_loss = encoder_layer(x)  # x: patch_num * batch * (2*d_model), 2 is for mean and std parts
            aux_loss_list.append(aux_loss)
        aux_loss = self.calculate_aus_loss( torch.tensor(aux_loss_list, requires_grad=True) )

        # re-parameterization trick
        x_mu = x[:,:, :self.d_model]
        x_std = self.softplus(x[:,:, self.d_model:])
        x = self.re_parameterization( x_mu, x_std )                # x: patch_num * batch * d_model

        x = self.output_layer(x, self.patch_layer.data_norm_mean, self.patch_layer.data_norm_std)   # x: seq_len * batch * d_out
        # output --- x: seq_len * batch * d_out
        return x, aux_loss, x_mu, x_std, mask_data

    def calculate_aus_loss(self, aux_loss_list):
        # consider how to calculate the aux_loss in a better way !!!
        aux_loss = aux_loss_list.mean()
        return aux_loss
    
    def patch_masking(self, data, rate):
        mask_index = torch.tensor(np.array([np.random.choice(data.shape[0], int(data.shape[0]*rate+0.5), replace=False) for i in range(data.shape[1])])).unsqueeze(dim=-1).repeat(1,1,data.shape[-1]).to(self.device)
        mask_value = torch.zeros_like(data.permute(1,0,2)).to(self.device)
        mask_data = data.permute(1,0,2).scatter(dim=1, index=mask_index, src=mask_value).permute(1,0,2)
        return mask_data
    
    def re_parameterization(self, mu, std):
        epsilon = torch.randn_like(std)
        sample_data = mu + std * epsilon
        return sample_data



class DataPatching(nn.Module):
    """
    Input: seq_len * batch * d_in
    Output: patch_num * batch * d_patch
    """
    def __init__(self, seq_len, d_in, d_patch):
        super(DataPatching, self).__init__()
        self.d_in = d_in
        self.d_patch = d_patch
        self.instance_norm = nn.InstanceNorm1d(seq_len)

    def forward(self, x):
        # input --- x: seq_len * batch * d_in
        # x = self.instance_norm(x.permute(1,2,0))    # x: batch * d_in * seq_len
        # x = x.permute(2,0,1)                        # x: seq_len * batch * d_in
        x = self.self_instance_norm(x)              # x: seq_len * batch * d_in
        x = self.patch_divide(x)                    # x: patch_num * batch * d_patch
        # output --- x: patch_num * batch * d_patch
        return x
    
    def patch_divide(self, data):
        split_data = data.transpose(2,0).split(self.d_patch, dim=-1)
        patch_data = torch.cat(split_data, dim=0)
        return patch_data
    
    def self_instance_norm(self, data, eps=1e-5):
        self.data_norm_mean = data.mean(dim=0, keepdim=True)
        self.data_norm_std = data.std(dim=0, keepdim=True, unbiased=False)
        data_norm = (data - self.data_norm_mean) / (self.data_norm_std + eps)
        return data_norm



class EmbeddingPE(nn.Module):
    """
    Input: patch_num * batch * d_patch
    Output: patch_num * batch * d_model
    """
    def __init__(self, d_patch, d_model, dropout, device):
        super(EmbeddingPE, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(d_patch, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, x):
        # input --- x: patch_num * batch * d_patch
        # note the input data dimension !!!
        seq_len = x.shape[0]   
        pe = self.position_encoding(seq_len).to(self.device)    # pe: patch_num * 1 * d_model
        x = self.embedding(x)                   # x: patch_num * batch * d_model
        # following the "attention is all you need" paper
        x = self.dropout(x + pe)                # x: patch_num * batch * d_model
        # output --- x: patch_num * batch * d_model
        return x

    def position_encoding(self, length):
        encoding = torch.zeros(length, self.d_model).unsqueeze(1)   # encoding: patch_num * 1 * d_model
        position = torch.arange(length).unsqueeze(1).unsqueeze(1)   # encoding: patch_num * 1 * 1
        encoding[:, :, 0::2] = torch.sin( position / torch.pow( 10000, torch.arange(0, self.d_model, 2, dtype=torch.float32)/self.d_model ) )
        encoding[:, :, 1::2] = torch.cos( position / torch.pow( 10000, torch.arange(1, self.d_model, 2, dtype=torch.float32)/self.d_model ) )
        # output --- encoding: patch_num * 1 * d_model
        return encoding



class DecoderOutput(nn.Module):
    def __init__(self, seq_len, num_patch, d_patch, d_model, d_out, dropout):
        super(DecoderOutput, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(d_model, d_patch)
        self.output = nn.Linear(num_patch*d_patch, seq_len*d_out)   # note the linear definition
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mean, x_std):
        # input --- x: patch_num * batch * d_model
        x = self.dropout(x.transpose(1,0))      # x: batch * patch_num * d_model
        x = self.activation(self.fc(x))         # x: batch * patch_num * patch_dim
        x = self.flatten(x)                     # x: batch * (patch_num * patch_dim)
        x = self.output(x)                      # x: batch * seq_len
        x = x.unsqueeze(-1).transpose(1,0)      # x: seq_len * batch * d_output
        x = self.inverse_instance_norm(x, x_mean, x_std)    # x: seq_len * batch * d_output
        # output --- x: seq_len * batch * d_output
        return x

    def inverse_instance_norm(self, data, mean, std, eps=1e-5):
        inverse_data = data * (std + eps) + mean
        return inverse_data
    


class MoEEncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, d_qkv, num_head, num_expert, dropout, top_k, training_flag, importance_factor, load_factor, device):
        super(MoEEncoderLayer, self).__init__()

        self.attention_1 = MultiHeadSelfAttention(d_model, d_qkv, num_head)
        self.attention_2 = MultiHeadSelfAttention(d_model, d_qkv, num_head)
        self.moe = MixtureOfExpert(d_model, d_hidden, num_expert, top_k, dropout, training_flag, importance_factor, load_factor, device)
        # self.moe = PositionwiseFeedForward_moe(d_model, d_hidden, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.add_norm_1 = AddNorm(d_model)
        self.add_norm_2 = AddNorm(d_model)
        self.add_norm_3 = AddNorm(d_model)
        self.add_norm_4 = AddNorm(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.dropout_4 = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        x = self.attention_1(x, x, x)
        x = self.dropout_1(x)
        x = self.add_norm_1(x, residual)

        residual = x
        x, aux_loss = self.moe(x)
        x = self.dropout_2(x)
        x = self.add_norm_2(x, residual)

        residual = x
        x = self.attention_2(x, x, x)
        x = self.dropout_3(x)
        x = self.add_norm_3(x, residual)

        residual = x
        x = self.ffn(x)
        x = self.dropout_4(x)
        x = self.add_norm_4(x, residual)

        return x, aux_loss



class MultiHeadSelfAttention(nn.Module):
    """
    Input: patch_num * batch * d_model
    Output: patch_num * batch * d_model
    """
    def __init__(self, d_model, d_qkv, num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head        # d_model = num_head * d_qkv
        self.W_q = nn.Linear(d_model, self.num_head*d_qkv)
        self.W_k = nn.Linear(d_model, self.num_head*d_qkv)
        self.W_v = nn.Linear(d_model, self.num_head*d_qkv)
        self.W_concat = nn.Linear(self.num_head*d_qkv, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # input --- query, key, value: patch_num * batch * d_model
        seq_len = key.shape[0]
        queries = torch.cat( self.W_q(query).chunk(self.num_head, dim=-1), dim=0 )  # queries: (num_head * patch_num) * batch * d_qkv
        keys = torch.cat( self.W_k(key).chunk(self.num_head, dim=-1), dim=0 )       # keys: (num_head * patch_num) * batch * d_qkv
        values = torch.cat( self.W_v(value).chunk(self.num_head, dim=-1), dim=0 )   # values: (num_head * patch_num) * batch * d_qkv

        similarity = torch.bmm( queries, keys.transpose(2,1) ) / np.sqrt(seq_len)   # similarity: (num_head * patch_num) * batch * batch
        # note the attention score, whether to output as record !!!
        self.attention_score = self.softmax(similarity)
        attention_results = torch.bmm(self.attention_score, values)                 # attention_results: (num_head * patch_num) * batch * d_qkv

        x = self.W_concat( torch.cat( attention_results.chunk(self.num_head, dim=0), dim=-1 ) )  # x: patch_num * batch * d_model
        # output --- x: patch_num * batch * d_model
        return x



class AddNorm(nn.Module):
    """
    Input: patch_num * batch * d_model
    Output: patch_num * batch * d_model
    """
    def __init__(self, d_model):
        super(AddNorm, self).__init__()
        # whether to choose batchnorm instead of layernorm, check it from the paper !!!
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        # input --- patch_num * batch * d_model
        x = self.layer_norm(x+residual)
        # output --- patch_num * batch * d_model
        return x



class PositionwiseFeedForward(nn.Module):
    """
    Input: patch_num * batch * d_model
    Output: patch_num * batch * d_model
    """
    def __init__(self, d_model, d_hidden, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input --- patch_num * batch * d_model
        x = self.activation( self.linear_1(x) )
        x = self.linear_2( self.dropout(x) )
        # output --- patch_num * batch * d_model
        return x


class PositionwiseFeedForward_moe(nn.Module):
    """
    Input: patch_num * batch * d_model
    Output: patch_num * batch * d_model
    """
    def __init__(self, d_model, d_hidden, dropout):
        super(PositionwiseFeedForward_moe, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input --- patch_num * batch * d_model
        x = self.activation( self.linear_1(x) )
        x = self.linear_2( self.dropout(x) )
        # output --- patch_num * batch * d_model
        return x, torch.tensor([0.0], requires_grad=True)


class MixtureOfExpert(nn.Module):
    """
    Input: patch_num * batch * d_model
    Output: patch_num * batch * d_model
    """
    def __init__(self, d_model, d_hidden, num_expert, top_k, dropout, training, importance_factor, load_factor, device):
        super(MixtureOfExpert, self).__init__()
        self.d_model = d_model
        self.hidden_size = d_hidden
        self.num_expert = num_expert
        self.top_k = top_k
        self.training = training
        self.importance_factor = importance_factor
        self.load_factor = load_factor

        self.experts = nn.ModuleList([ PositionwiseFeedForward(d_model, d_hidden, dropout) for _ in range(num_expert) ])
        self.W_gate = nn.Parameter(torch.zeros(d_model, num_expert), requires_grad=True)
        self.W_noise = nn.Parameter(torch.zeros(d_model, num_expert), requires_grad=True)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

        self.device = device
    
    def forward(self, x):
        # input --- patch_num * batch * d_model
        gates, loads = self.top_k_gate_noisy(x)                                                 # gates / loads: patch_num * batch * expert_num
        aux_loss = self.aux_loss(gates, loads, self.importance_factor, self.load_factor)        # aux_loss: float value
        
        dispatcher = SparseDispatcher(gates, self.num_expert, self.device)                                   # dispatcher: an instance of the class

        expert_inputs = dispatcher.dispatch(x)                                                  # expert_inputs: 
        expert_outputs = [self.experts[i](expert_inputs[i].to(self.device)) for i in range(self.num_expert)]
        x = dispatcher.combine(expert_outputs)

        return x, aux_loss
    
    def top_k_gate_noisy(self, x, noise_epsilon=1e-2):
        # input --- patch_num * batch * d_model
        # use @ instead of torch.bmm, because two elements are not with the shape dimension
        clean_values = x @ self.W_gate                                                  # clean_values: patch_num * batch * expert_num

        if self.training:
            noise_std = self.softplus( x @ self.W_noise ) + noise_epsilon               # noise_std: patch_num * batch * expert_num
            noisy_values = clean_values + torch.rand_like(clean_values) * noise_std     # noisy_values: patch_num * batch * expert_num
            values = noisy_values
        else:
            values = clean_values
        # select top k+1 rather than top k, which facilitates the prob_k_load operation
        top_values, top_indices = values.topk( min(self.top_k+1, self.num_expert) )     # top_values / top_indices: patch_num * batch * (k+1)
        top_k_values = top_values[:, :, :self.top_k]                                    # top_k_values: patch_num * batch * k
        top_k_indices = top_indices[:, :, :self.top_k]                                  # top_k_indices: patch_num * batch * k

        top_k_gates = self.softmax(top_k_values)                                        # top_k_gates: patch_num * batch * k
        raw_gates = torch.zeros_like(values, requires_grad=True)                        # raw_gates: patch_num * batch * expert_num
        # fill in the top_k_gates into raw_gates according to the index from top_k_indices
        gates = raw_gates.scatter(-1, top_k_indices, top_k_gates)                       # gates: patch_num * batch * expert_num

        if self.training:
            loads = self.prob_in_top_k(clean_values, noisy_values, noise_std, top_values)   # loads: patch_num * batch * expert_num
        else:
            loads = (gates > 0).sum(0)
        return gates, loads

    def prob_in_top_k(self, clean_values, noisy_values, noise_std, top_values):
        top_values_flatten = top_values.flatten()                                                                                       # top_values_flatten: (patch_num * batch * (k+1))
        # check the device !!!
        top_k_threshold_position = (torch.arange( clean_values.shape[0]*clean_values.shape[1] ) * top_values.shape[-1] + self.top_k).to(self.device)      # top_k_threshold_position: (patch_num * batch)
        # select the top k values exclude itself (need to consider top k+1 values)
        top_k_threshold_value = torch.unsqueeze( torch.gather(top_values_flatten, 0, top_k_threshold_position), 1 )                    # top_k_threshold_value: (patch_num * batch) * 1
        out_of_threshold_position = top_k_threshold_position - 1                                                                        # out_of_threshold_position: (patch_num * batch)
        out_of_threshold_value = torch.unsqueeze( torch.gather(top_values_flatten, 0, out_of_threshold_position), 1 )                   # out_of_threshold_value: (patch_num * batch) * 1
        noise_in_top_k = torch.gt( noisy_values.reshape(clean_values.shape[0]*clean_values.shape[1], -1), top_k_threshold_value )       # noise_in_top_k: (patch_num * batch) * expert_num

        # follow the equation in original paper, by using standard normal distribution
        normal_distribution = torch.distributions.normal.Normal(self.mean, self.std)
        prob_top_k = normal_distribution.cdf( (clean_values.reshape(clean_values.shape[0]*clean_values.shape[1], -1) - top_k_threshold_value) / noise_std.reshape(clean_values.shape[0]*clean_values.shape[1], -1) )                                                                                                                           # prob_top_k: (patch_num * batch) * expert_num
        prob_out_of_threshold = normal_distribution.cdf( (clean_values.reshape(clean_values.shape[0]*clean_values.shape[1],-1) - out_of_threshold_value) / noise_std.reshape(clean_values.shape[0]*clean_values.shape[1], -1) )
        prob = torch.where(noise_in_top_k, prob_top_k, prob_out_of_threshold).reshape(clean_values.shape[0], clean_values.shape[1], -1) # prob: patch_num * batch * expert_num
        # output --- patch_num * batch * expert_num
        return prob

    def aux_loss(self, gates, loads, importance_factor, load_factor, eps=1e-10):
        # summation in both seq_len and batch dimensions
        expert_importance = gates.sum(dim=0).sum(dim=0)
        expert_load = loads.sum(dim=0).sum(dim=0)
        # the square of the coefﬁcient of variation, i.e., cv^2 = variance / mean^2
        importance_balance = expert_importance.float().var() / (expert_importance.float().mean()**2 + eps)
        load_balance = expert_load.float().var() / (expert_load.float().mean()**2 + eps)
        
        aux_loss = importance_factor * importance_balance + load_factor * load_balance
        # output --- float value
        return aux_loss



class SparseDispatcher(object):
    def __init__(self, gates, num_expert, device):
        self.device = device
        self.gates = gates
        self.num_expert = num_expert
        # sort by column, and the last column corresponds the expert number (0,1,...)
        sorted_expert, sorted_index = torch.nonzero(gates).sort(dim=0)          # sorted_expert / sorted_index: (patch_num * batch * expert_num) * 3
        
        # split by column, and the value is in ascending order (shown as group by expert)
        _, _, self.expert_index = sorted_expert.split(1, dim=1)                 # self.expert_index: (patch_num * batch * expert_num) * 1

        # according to the index of expert in gates (i.e., sorted_index[:, 2]), and then find the corresponding indexes of sequence and batch in gates
        self.sequence_index = torch.nonzero(gates)[ sorted_index[:, 2], 0 ].to(self.device)     # self.sequence_index: (patch_num * batch * expert_num)
        self.batch_index = torch.nonzero(gates)[ sorted_index[:, 2], 1 ].to(self.device)        # self.batch_index: (patch_num * batch * expert_num)

        self.expert_count = (gates > 0).sum(0).sum(0).tolist()                  # self.expert_count: expert_num
        
        gates_expand = torch.tensor( [ self.gates[self.sequence_index[i]][self.batch_index[i]].tolist() for i in range(self.sequence_index.shape[0]) ] ).to(device)
        self.nonzero_gates = torch.gather(gates_expand, 0, self.expert_index)   # self.nonzero_gates: (patch_num * batch * expert_num) * expert_num
    
    def dispatch(self, x):
        # select inputs of each expert from raw input according to the sequence and batch dimension
        all_inputs = torch.tensor( [ x[self.sequence_index[i]][self.batch_index[i]].tolist() for i in range(self.sequence_index.shape[0]) ] )
        input_by_expert = torch.split(all_inputs, self.expert_count, dim=0)     # input_by_expert: expert_num (list)
        return input_by_expert
    
    def combine(self, x, weighted=True):
        expert_output = torch.cat(x, 0)
        if weighted:
            expert_output = expert_output.mul(self.nonzero_gates)
        all_output = torch.zeros( self.gates.shape[0], self.gates.shape[1], x[-1].shape[-1] ).to(self.device)
        all_output = self.self_index_add(all_output, 0, self.sequence_index, self.batch_index, expert_output.float())
        return all_output

    def self_index_add(self, target, dim, sequence_index, batch_index, source):
        if sequence_index.shape[0] == batch_index.shape[0] and dim == 0:
            for i in range(sequence_index.shape[0]):
                target[sequence_index[i], batch_index[i], :] += source[i]
        return target