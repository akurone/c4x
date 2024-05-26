# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import torch
import torch.nn as nn
import numpy as np
from transformers.modeling_utils import PreTrainedModel  #, Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from decision_mamba_layers import Block


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])  #, scale=True
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class DecisionMamba(PreTrainedModel):
    config_class = GPT2Config
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.config = config
        self.state_dim = config.state_dim
        self.act_dim = config.act_dim
        self.hidden_size = config.n_embd
        self.max_length = config.n_positions

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(self.config)
        self.remove_act_embs = config.remove_act_embs

        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # time embeddings are treated similar to positional embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings
        if not self.remove_act_embs:
            action_embeddings = self.embed_action(actions) + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.remove_act_embs:
            num_token_type = 2
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
        else:
            num_token_type = 3
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, num_token_type*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = self.transformer(inputs_embeds=stacked_inputs)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, num_token_type, self.hidden_size).permute(0, 2, 1, 3)

        state_reps = x[:,1]
        action_preds = self.predict_action(state_reps)  # predict next action given state
        return action_preds

class TrainableDM(DecisionMamba):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        # add the DT loss
        action_preds = super().forward(kwargs["states"], kwargs["actions"], kwargs["returns_to_go"], kwargs["timesteps"])
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        # Use Cross Entropy Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(action_preds, action_targets)

        return {"loss": loss}
    
    def original_forward(self, **kwargs):
        return super().forward(kwargs["states"], kwargs["actions"], kwargs["returns_to_go"], kwargs["timesteps"])


# Constants represent the position of the value within a trajectory-array
# trajectory structure in the dataset: traj = [length, states, actions, rewards, RTGS, dones]
LENGTH, STATES, ACTIONS, REWARDS, RTGS, DONES = 0, 1, 2, 3, 4, 5
class DecisionMambaGymDataCollator:
    return_tensors: str = "pt"
    max_len: int = 21 #subsets of the episode we use for training
    state_dim: int = 42  # size of state space
    act_dim: int = 7  # size of action space
    max_ep_len: int = 42 # max episode length in the dataset TODO: is this the correct value?
    #scale: float = 1000.0  # normalization of rewards/returns

    def __init__(self, dataset) -> None:
        self.act_dim = 7
        self.state_dim = 42
        self.dataset = dataset
        self.n_traj = len(dataset)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(np.arange(self.n_traj), size=batch_size, replace=True)
        
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
        for ind in batch_inds:
            # Select trajectory at given index
            feature = self.dataset[int(ind)]
            #set si randomly to start somewhere in the sequence (but at least max_len steps before the end..)
            length = self.dataset[ind][LENGTH]
            if length <= self.max_len:
                si = 0
            else:
                # we should just start from the end then, because we have sparse rewards...
                #si = random.randint(0, length - self.max_len)
                #NOTE: this case should never occur anyway if we work with a window size of the maximum possible episode length..
                si = max(0, length - self.max_len - 1)

            # get sequences from dataset
            s.append(np.array(feature[STATES][si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature[ACTIONS][si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature[REWARDS][si : si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(feature[DONES][si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature[REWARDS][si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        # Converting
        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        # This is how the trajectories are returned for the transformer to learn with them
        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }