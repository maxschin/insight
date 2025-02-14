import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.distributions.normal import Normal
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

#cnn
from . import Normal_Cnn

# eql
from .eql import functions
from .eql.symbolic_network import SymbolicNet, SymbolicNetSimplified

import copy

from transform_data import oca_obj_to_cnn_coords

def load_agent(class_name):
    """
    Return the agent class matching the given class name.
    
    Parameters:
        class_name (str): The name of the agent class.
        
    Returns:
        class: The agent class if found.
        
    Raises:
        ValueError: If the class name is not found in the module.
    """
    agent_class = globals().get(class_name)
    if agent_class is None:
        raise ValueError(f"Agent class '{class_name}' not found in module '{__name__}'.")
    return agent_class

class Agent(nn.Module):
    def __init__(self, envs, args,nnagent=None, agent_in_dim=None, skip_perception=False):
        super().__init__()
        if not skip_perception and args.gray:
            self.network = Normal_Cnn.OD_frames_gray2(args)
        elif not skip_perception:
            self.network = Normal_Cnn.OD_frames(args)
        self.args = args
        self.skip_perception = skip_perception
        self.activation_funcs = [
            *[functions.Pow(2)] * 2 * args.n_funcs,
            *[functions.Pow(3)] * 2 *args.n_funcs,
            *[functions.Constant()] * 2 * args.n_funcs,
            *[functions.Identity()] * 2 * args.n_funcs,
            *[functions.Product()] * 2 * args.n_funcs,
            *[functions.Add()] * 2 * args.n_funcs,]
        self.eql_actor = SymbolicNet(
            args.n_layers,
            funcs=self.activation_funcs,
            in_dim=agent_in_dim if self.skip_perception else args.cnn_out_dim,
            out_dim=envs.single_action_space.n)
        self.eql_inv_temperature = 10
        self.neural_actor = nn.Sequential(
            nn.Linear(2048 if not self.skip_perception else agent_in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, envs.single_action_space.n))
        self.critic = nn.Sequential(
            nn.Linear(2048 if not self.skip_perception else agent_in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        if not skip_perception and args.load_cnn:
            print('load cnn')
            self.network = torch.load('models/'+f'{args.env_id}'+f'{args.resolution}'+f'{args.obj_vec_length}'+f"_gray{args.gray}"+f"_objs{args.n_objects}"+f"_seed{args.seed}"+'_od.pkl')
        self.deter_action = args.deter_action
        self.nnagent= nnagent
        if self.nnagent:
            self.critic = copy.deepcopy(self.nnagent.critic)
            self.network = copy.deepcopy(self.nnagent.network)

    def get_value(self, x):
        if self.skip_perception:
            batch_size = x.shape[0]
            hidden = x.reshape(batch_size, -1) / 255.0
        else:
            hidden = self.network.encoder(x / 255.0)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, threshold=0.8, actor="neural"):
        if actor == "neural":
            if self.skip_perception:
                batch_size = x.shape[0]
                hidden = x.reshape(batch_size, -1) / 255.0
            else:
                hidden = self.network.encoder(x / 255.0)
            logits = self.neural_actor(hidden) 
        else:
            # Using ocatari object data instead of CNN coordinates for eql actor
            batch_size = x.shape[0]
            hidden = x.reshape(batch_size, -1) / 255.0
            logits = self.eql_actor(hidden) * self.eql_inv_temperature
        
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden), logits, dist.probs
    
    def get_pretrained_action_and_value(self, x, action=None):
        if self.skip_perception:
            raise NotImplementedError("When skip_perception is True, pretrained cannot be used (yet)")
        hidden = self.nnagent.network(x/255.0)
        logits_nn = self.nnagent.actor(hidden)
        probs_nn = Categorical(logits=logits_nn)
        if action is None:
            action = probs_nn.sample()
        return action, probs_nn.log_prob(action), probs_nn.entropy(), self.nnagent.critic(hidden),logits_nn,probs_nn.probs

class AgentSimplified(Agent):
    def __init__(self, envs, args,nnagent=None, agent_in_dim=None, skip_perception=True):
        super().__init__(envs, args, nnagent, agent_in_dim, skip_perception=skip_perception)
        self.activation_funcs = [
            functions.Pow(2),
            functions.Pow(3),
            functions.Identity(),
            functions.Product()
        ]
        # extend init
        self.eql_actor = SymbolicNetSimplified(
            funcs=self.activation_funcs,
            in_dim=agent_in_dim if self.skip_perception else args.cnn_out_dim,
            out_dim=envs.single_action_space.n)

class AgentContinues(nn.Module):
    def __init__(self, envs, args,nnagent=None):
        super().__init__()
        if args.gray:
            self.network = Normal_Cnn.OD_frames_gray2(args)
        else:
            self.network = Normal_Cnn.OD_frames(args)
        self.args = args
        self.activation_funcs = [
            *[functions.Pow(2)] * 2 * args.n_funcs,
            *[functions.Pow(3)] * 2 * args.n_funcs,
            *[functions.Constant()] * 2 * args.n_funcs,
            *[functions.Identity()] * 2 * args.n_funcs,
            *[functions.Product()] * 2 * args.n_funcs,
            *[functions.Add()] * 2 * args.n_funcs,]
        self.eql_actor = SymbolicNet(
            args.n_layers,
            funcs=self.activation_funcs,
            in_dim=args.cnn_out_dim+args.ego_state_dim*args.ego_state,
            out_dim=np.prod(envs.single_action_space.shape))
        self.eql_actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.eql_inv_temperature = 10
        self.action_space = np.prod(envs.single_action_space.shape)
        self.neural_actor = nn.Sequential(
            nn.Linear(args.cnn_out_dim+args.ego_state_dim*args.ego_state, 512),
            nn.ReLU(),
            nn.Linear(512, np.prod(envs.single_action_space.shape)))
        self.neural_actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.ego_stste_normalizer = nn.LayerNorm(args.ego_state_dim)
        self.critic = nn.Sequential(
            nn.Linear(args.cnn_out_dim+args.ego_state_dim*args.ego_state, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        if args.load_cnn:
            print('load cnn')
            cnn_ckpt = ( 'models/'+f'{args.env_id}{args.resolution}'
                        f'{args.obj_vec_length}_gray{args.gray}'
                        f'_objs{args.n_objects}_seed{args.seed}_od.pkl')
            self.network = torch.load(cnn_ckpt)
        self.nnagent= nnagent
        if self.nnagent:
            self.critic = copy.deepcopy(self.nnagent.critic)
            self.network = copy.deepcopy(self.nnagent.network)
        self.action_dist = SquashedDiagGaussianDistribution(
            np.prod(envs.single_action_space.shape))

    def get_value(self, x, next_state=None):
        hidden = self.network.encoder(x / 255.0)
        if self.args.ego_state:
            hidden = torch.concat((next_state,hidden),dim=-1)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, threshold=0.8, actor="neural", next_state=None, deterministic=False):
        hidden = self.network.encoder(x / 255.0)
        if not next_state is None:
            next_state = self.ego_stste_normalizer(next_state)
        if self.args.ego_state:
            hidden = torch.concat((next_state,hidden),dim=-1)
        if actor == "neural":
            action_mean = self.neural_actor(hidden) 
            action_logstd = self.neural_actor_logstd
        else:
            coordinates = self.network(x / 255.0, threshold=threshold)
            hidden_state = coordinates
            if self.args.ego_state:
                hidden_state = torch.concat((next_state,coordinates),dim=-1)
            action_mean = self.eql_actor(hidden_state) #* self.eql_inv_temperature
            action_logstd = self.eql_actor_logstd
            
        if action is None:
            if deterministic:
                self.action_dist.proba_distribution(action_mean, action_logstd)
                action = self.action_dist.mode()
                action = action.detach()
                log_prob = self.action_dist.log_prob(action)
            else:
                action, log_prob = self.action_dist.log_prob_from_params(
                    action_mean, action_logstd)
                action = action.detach()
        else:
            self.action_dist.proba_distribution(action_mean, action_logstd)
            action = action.detach()
            log_prob = self.action_dist.log_prob(action)
        entropy = -log_prob.mean()
        prob = torch.exp(log_prob)
        value = self.critic(hidden)
        # return action, log_prob, entropy, value, logits, prob
        return action, log_prob, entropy, value, action_mean, prob 
    
    def get_pretrained_action_and_value(self, x, action=None, next_state=None, deterministic=False):
        return self.get_action_and_value(
            x=x, action=action, actor="neural", next_state=next_state, deterministic=deterministic)
