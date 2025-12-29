import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from trajectory_model import TrajectoryModel
from trajectory_gpt2 import GPT2Model
from img_encoder import Encoder

import iq
from utils import pad


class DecEncoder(TrajectoryModel):

    """
    This model uses GPT to model (Lang, state_1, action_1, state_2, ...) or (state_1, option_1, action_1, ...)
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            option_dim,
            lang_dim,
            discrete,
            hidden_size,
            K=10,
            use_language=False,
            use_options=True,
            option_il=False,
            predict_q=False,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            no_states=False,
            no_actions=False,
            embed_state=None,
            ** kwargs):
        # max_length used to be K
        super().__init__(state_dim, action_dim, max_length=max_length)

        self.use_options = use_options
        self.use_language = use_language
        self.option_il = option_il
        self.predict_q = predict_q

        if use_language and use_options:
            raise ValueError("Cannot use language and options!")
        if not use_language and not use_options:
            raise ValueError("Have to use language or options!")
        self.option_dim = option_dim
        self.discrete = discrete

        self.hidden_size = hidden_size

        # if isinstance(state_dim, tuple):
        #     # LORL
        #     if state_dim[0] == 3:
        #         # LORL Sawyer
        #         self.embed_state = Encoder(hidden_size=hidden_size, ch=3, robot=False)
        #     else:
        #         # LORL Franka
        #         self.embed_state = Encoder(hidden_size=hidden_size, ch=12, robot=True)
        # else:
        #     self.embed_state = nn.Linear(self.state_dim, hidden_size)
        # self.embed_state = embed_state  # TODO

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # self.transformer = GPT2Model(config) TODO
        # self.diffusion_model = diffusion_model

        # self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # self.embed_action = nn.Linear(self.act_dim, hidden_size)

        self.no_states = no_states
        self.no_actions = no_actions

        # if use_options:
        #     self.embed_option = nn.Linear(self.option_dim, hidden_size)  # TODO  self.option_dim == hidden_size

        # if use_language:
        #     self.embed_lang = nn.Linear(lang_dim, hidden_size)

        # self.embed_ln = nn.LayerNorm(hidden_size  + self.act_dim)  # TODO
        self.embed_ln = nn.LayerNorm(hidden_size)
        # if kwargs['eval'] == True:
        #     self.embed_ln = nn.LayerNorm(hidden_size  + self.act_dim)
        # else:
        #     self.embed_ln = nn.LayerNorm(hidden_size)  # TODO
        # self.embed_bn = nn.BatchNorm1d(K)
        # note: we don't predict states or returns for the paper
        if isinstance(self.state_dim, int):
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh and not discrete else []))
        # )
        # if use_options:
        #     self.predict_option = torch.nn.Linear(hidden_size, self.option_dim)  # TODO
        if predict_q:
            self.predict_q = torch.nn.Linear(hidden_size, self.act_dim)

    def encode(self, actions, options=None, state_embeddings=None):
        assert options is not None
        option_embeddings = options  # TODO  no operation here
        option_embeddings = torch.mean(option_embeddings, dim=1)

        if self.embed_ln.normalized_shape[-1] == self.hidden_size:
            state_embeddings = self.embed_ln(state_embeddings)
            stacked_inputs = torch.cat((actions, state_embeddings), dim=2)  

        else:
            stacked_inputs = self.embed_ln(torch.cat((actions, state_embeddings), dim=2)) 
            # state_embeddings = self.embed_ln(state_embeddings)
            # stacked_inputs = torch.cat((actions, state_embeddings), dim=2)   # (B, seq_length, )

        outputs = {'stacked_inputs': stacked_inputs,
                    'option_embeddings': option_embeddings,
                    }
        return outputs


    def decode(self, traj_out):
        if self.use_options:
            action_preds = traj_out[:, :, :self.act_dim]
            outputs = {'action_preds': action_preds,}

            if self.predict_q:
                # predict next Q given state and option   ## IMP: Don't use current action
                q_preds = self.predict_q(traj_out[:, :, self.act_dim:])
                outputs.update({'q_preds': q_preds})

            return outputs

    def get_action(self, states, actions, timesteps, options=None, embed_state=None, word_embeddings=None, **kwargs):

        if self.use_options:
            assert options is not None
            if isinstance(self.state_dim, tuple):
                states = states.reshape(1, -1, *self.state_dim)
            else:
                states = states.reshape(1, -1, self.state_dim)
            options = options.reshape(1, -1, self.option_dim)
            actions = actions.reshape(1, -1, self.act_dim)
            timesteps = timesteps.reshape(1, -1)

            # self.max_length = 32
            if self.max_length is not None:
                states = states[:, -self.max_length:]
                options = options[:, -self.max_length:]
                actions = actions[:, -self.max_length:]
                timesteps = timesteps[:, -self.max_length:]

                # pad all tokens to sequence length  TODO Important
                # attention_mask = pad(torch.ones(1, states.shape[1]), self.max_length).to(
                #     dtype=torch.long, device=states.device).reshape(1, -1)
                states = pad(states, self.max_length).to(dtype=torch.float32)
                options = pad(options, self.max_length).to(dtype=torch.float32)
                actions = pad(actions, self.max_length).to(dtype=torch.float32)
                timesteps = pad(timesteps, self.max_length).to(dtype=torch.long)

                states = states.to(dtype=torch.float32)
                options = options.to(dtype=torch.float32)
                actions = actions.to(dtype=torch.float32)
                timesteps = timesteps.to(dtype=torch.long)
            else:
                raise ValueError('Should not have max_length NONE')

            attention_mask = None

            with torch.no_grad():
                state_embeddings = embed_state(states)
                encoder_out = self.encode(states, actions, timesteps, options=options,
                                          state_embeddings=state_embeddings,
                                          attention_mask=attention_mask)

            # preds = self.forward(
            #     states, actions, timesteps, options=options, attention_mask=attention_mask, get_action=True)

            return encoder_out

        if self.use_language:
            assert word_embeddings is not None
            if isinstance(self.state_dim, tuple):
                states = states.reshape(1, -1, *self.state_dim)
            else:
                states = states.reshape(1, -1, self.state_dim)
            actions = actions.reshape(1, -1, self.act_dim)
            timesteps = timesteps.reshape(1, -1)

            if self.max_length is not None:
                states = states[:, -self.max_length:]
                actions = actions[:, -self.max_length:]
                timesteps = timesteps[:, -self.max_length:]

                # pad all tokens to sequence length
                attention_mask = pad(
                    torch.ones(1, states.shape[1]),
                    self.max_length).to(
                    dtype=torch.long, device=states.device).reshape(
                    1, -1)
                states = pad(states, self.max_length).to(dtype=torch.float32)
                actions = pad(actions, self.max_length).to(dtype=torch.float32)
                timesteps = pad(timesteps, self.max_length).to(dtype=torch.long)
            else:
                attention_mask = None

            preds = self.forward(
                states, actions, timesteps, word_embeddings=word_embeddings, attention_mask=attention_mask, **kwargs)

            return preds
