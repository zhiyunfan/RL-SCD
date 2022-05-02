import torch
from torch import nn
import numpy as np
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
# from utils.tensor_functions import compute_in_batches

from graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
# from utils.beam_search import CachedLookup
# from utils.functions import sample_many
from torch.distributions import Categorical
from metric.segmentation import SegmentationPurityCoverageFMeasure
metric = SegmentationPurityCoverageFMeasure()


class ScdStateMaker(nn.Module):
    def __init__(self, ori_states):
        super(ScdStateMaker, self).__init__()
        self.ori_states = ori_states
        self.mean_state = torch.mean(self.ori_states[:, :, :-1], axis=1)

    def get_cur_state(self, cur_time_step, last_act, last_rewards):
        cur_state = self.ori_states[:, cur_time_step]
        last_action = last_act
        last_action_one_hot = np.where(last_action == 1, [0, 1], [1, 0])  # bs.2
        rewards = last_rewards
        self.global_state = np.concatenate((cur_state, last_action_one_hot, rewards, rewards), axis=1)
        # state_all = np.concatenate((self.mean_state, self.global_state), axis=1)
        return torch.from_numpy(self.global_state).unsqueeze(1).to(torch.float32)

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )

class AttentionModel(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 episode_length,
                 batch_size=32,
                 n_encode_layers=3,
                 tanh_clipping=10.,
                 normalization='batch',
                 n_heads=8):
        super(AttentionModel, self).__init__()

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.glimpse_embedding_dim = 192
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.episode_length = episode_length

        self.step_context_dim = 2 * embedding_dim  # Embedding of first and last node

        self.tanh_clipping = tanh_clipping

        # self.mask_inner = mask_inner
        # self.mask_logits = mask_logits

        self.n_heads = n_heads
        # self.checkpoint_encoder = checkpoint_encoder
        # self.shrink_size = shrink_size

        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self._init_embed = nn.Linear(input_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * self.glimpse_embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(self.step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(self.glimpse_embedding_dim, self.glimpse_embedding_dim, bias=False)


    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)  # 1024.20.128
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_parallel_step_context(self, embeddings, state, timestep, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state
        batch_size, state_dim = current_node.size()
        _first_timestep = torch.zeros([batch_size, 1], dtype=torch.int64)
        _current_timestep = torch.full([batch_size, 1], timestep, dtype=torch.int64)
        if timestep == 0:
            # First and only step, ignore prev_a (this is a placeholder)
            return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
        else:
            return embeddings.gather(1, torch.cat((_first_timestep, _current_timestep), \
                                                  1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))).view(batch_size, 1, -1)

    def _get_attention_node_data(self, fixed):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, 1, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, 1, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, 1, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        # if self.mask_logits:
        #     logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_log_p(self, fixed, state, timestep, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, timestep))  # torch.Size([1024, 1, 128])

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
        # torch.Size([8, 64, 1, 400, 16]),  torch.Size([8, 64, 1, 400, 16]),  torch.Size([64, 1, 400, 128])

        # Compute logits (unnormalized log_p) 在这里只query下一时刻的点就行
        dynamic_state = self.scd_state_maker.get_cur_state(timestep, self.last_act, self.last_rewards)

        next_time_step = timestep+1 if timestep < self.episode_length-1 else timestep
        query = torch.cat((query, dynamic_state), dim=-1)

        logits, glimpse = self._one_to_many_logits(query, glimpse_K[:, :, :, [next_time_step]], glimpse_V[:, :, :, [next_time_step]], logit_K[:,:,[next_time_step]])
        # torch.Size([64, 1, 1]), torch.Size([64, 1, 128])
        logits = torch.nn.Sigmoid()(logits)

        logits_2 = torch.cat((1 - logits, logits), dim=-1)
        dis_ = Categorical(logits_2)
        action = dis_.sample()
        log_p = dis_.log_prob(action)

        assert not torch.isnan(log_p).any()

        return log_p, action

    def _inner_decode(self, input, embeddings):
        # torch.Size([50, 400, 60])
        log_p_s = []
        actions_s = []
        rewards = []
        label_lst = []
        p_lst, c_lst, f_lst = [], [], []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        self.scd_state_maker = ScdStateMaker(input)

        # Perform decoding steps
        i = 0
        peak_rewards = torch.full([self.bacth_size, 1], 10)
        while i < self.episode_length:
            print("i_step:", i)
            state = input[:, i]  ## torch.Size([64, 122])

            log_p, action = self._get_log_p(fixed, state, i)
            # Collect output of step
            log_p_s.append(log_p)
            actions_s.append(action.numpy())
            i += 1

            ### 计算 预测正确转换点的 rewards
            label = state[:, [-1]]
            label_lst.append(label.numpy())
            label_eq_act = (label == action)
            if_change = (action == 1)
            cur_peak_reward = peak_rewards * label_eq_act * if_change
            ### 计算 其他的 rewards 
            for j in range(self.bacth_size):
                p, c, f = metric.score2metric(np.array(actions_s)[:,j], np.array(label_lst)[:,j])
                p_lst.append([p])
                c_lst.append([c])
                f_lst.append([f])
            f_rewards = np.array(f_lst[-self.bacth_size:])
            f_rewards = torch.from_numpy(f_rewards)
            self.last_rewards = f_rewards + cur_peak_reward
            # self.last_rewards =  cur_peak_reward
            rewards.append(self.last_rewards)
            self.last_act = action

        return log_p_s, actions_s, rewards



    def forward(self, input):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        #### 初始化
        self.bacth_size = input.size()[0]
        self.last_act = torch.zeros(self.bacth_size, 1)
        self.last_rewards = torch.zeros(self.bacth_size, 1)

        embeddings, _ = self.embedder(self._init_embed(input))  # return:torch.Size([64, 400, 128])

        _log_p, action, rewards = self._inner_decode(input, embeddings)

        return torch.stack(_log_p, axis=1), torch.tensor(action).permute(1,0,2), torch.stack(rewards, axis=1)














