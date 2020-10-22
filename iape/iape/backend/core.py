import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


from .aws_rnn import RNNWDModel
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w

def max2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = int(np.floor((h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) / stride[0] + 1))
    w = int(np.floor((h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) / stride[1] + 1))

    return h, w

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


#Observation CNNs
class ResidualBlock(nn.Module):
    def __init__(self, use_batchnorm=False, use_layernorm=False, gate=F.relu, ch_out=16,hw=(28,28)):
        super(ResidualBlock, self).__init__()
        self.gate = gate
        self.conv1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=3 // 2)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=3 // 2)
        if use_batchnorm:
            self.reg1 = nn.BatchNorm2d(ch_out)
            self.reg2 = nn.BatchNorm2d(ch_out)
        elif use_layernorm:
            self.reg1 = nn.LayerNorm(ch_out)
            self.reg2 = nn.LayerNorm(ch_out)
        else:
            self.reg1 = nn.Identity()
            self.reg2 = nn.Identity()

        hw = conv_output_shape(hw,3,1,3//2)
        hw = conv_output_shape(hw,3,1,3//2)
        self.hw= hw

    def forward(self, x):
        y = self.gate(x)
        y = self.gate(self.reg1(self.conv1(y)))
        y = self.reg2(self.conv2(y))
        y = y + x
        return y

class MetaBlock(nn.Module):
    def __init__(self, use_batchnorm=False, use_layernorm=False, gate=F.relu, ch_in=16, ch_out=16,hw=(28,28)):
        super(MetaBlock, self).__init__()

        self.gate = F.relu
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=3 // 2)
        if use_batchnorm:
            self.reg1 = nn.BatchNorm2d(ch_out)
        elif use_layernorm:
            self.reg1 = nn.LayerNorm(ch_out)
        else:
            self.reg1 = nn.Identity()

        hw = conv_output_shape(hw, 3, 1, 3 // 2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        hw = max2d_output_shape(hw, 3, 2,)
        self.resblock1 = ResidualBlock(use_batchnorm=use_batchnorm, use_layernorm=use_layernorm, gate=gate,
                                       ch_out=ch_out, hw=hw)
        hw = self.resblock1.hw

        self.resblock2 = ResidualBlock(use_batchnorm=use_batchnorm, use_layernorm=use_layernorm, gate=gate,
                                       ch_out=ch_out, hw=hw)
        hw = self.resblock2.hw
        self.hw=hw

    def forward(self,x):
        y = self.conv1(x)
        y = self.pool(y)
        y = self.resblock1(y)
        y = self.resblock2(y)
        return y

class ImpalaCNN(nn.Module):

    def __init__(self, obs_size, use_batchnorm=True, use_layernorm=False, gate=F.elu ):
        super(ImpalaCNN, self).__init__()

        self.hw = obs_size[1:]
        ch_in = obs_size[0]
        self.gate=gate
        self.meta_block1=MetaBlock(use_batchnorm=use_batchnorm, use_layernorm=use_layernorm, gate=gate, ch_in=ch_in, ch_out=16,hw=self.hw)
        self.hw =self.meta_block1.hw
        self.meta_block2=MetaBlock(use_batchnorm=use_batchnorm, use_layernorm=use_layernorm, gate=gate, ch_in=16, ch_out=32,hw=self.hw)
        self.hw = self.meta_block2.hw
        self.meta_block3=MetaBlock(use_batchnorm=use_batchnorm, use_layernorm=use_layernorm, gate=gate, ch_in=32, ch_out=32,hw=self.hw)
        self.hw = self.meta_block3.hw
        self.linear = nn.Linear( self.hw[0]*self.hw[1]*32, 256)

    def forward(self,x):
        y = self.meta_block1(x)
        y = self.meta_block2(y)
        y = self.meta_block3(y)
        y = torch.flatten(y,start_dim=1)
        y= self.gate(y)
        y= self.gate(self.linear(y))

        return y

#Simpler CNN architecture
class VisualEncoder(nn.Module):
    def __init__(self, embedding_size=1024, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.reshape(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy(), pi.logits

    def act(self, obs):
        return self.step(obs)[0]

class CNNActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape

        self.pi = CnnCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = CnnCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), pi.logits.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class CnnCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()

        #roll HWC input to CWH
        obs_size = tuple(np.roll(obs_dim,1))
        self.v_feature_net = ImpalaCNN(obs_size)
        self.v_net = nn.Linear(256,1)


    def forward(self, obs):
        # expected image input is B x(??) x HWC, rearranging to (B??)CHW before forward and returning in B x ?? format
        obs_shape = obs.shape
        v = self.v_net(self.v_feature_net(obs.reshape(-1, *obs_shape[-3:]).transpose(-3,-1))).view(*obs_shape[:-3])  # Critical to ensure v has right shape.
        return v

class CnnCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        #roll HWC input to CWH
        obs_size = tuple(np.roll(obs_dim,1))
        self.logit_feature_net = ImpalaCNN(obs_size)
        self.logits_net = nn.Linear(256,act_dim)

    def _distribution(self, obs):
        # expected image input is B x(??) x HWC, rearranging to (B??)CHW before forward and returning in B x ?? format
        obs_shape = obs.shape
        logit_features = self.logit_feature_net(obs.reshape(-1, *obs_shape[-3:]).transpose(-3, -1)).view(*obs_shape[:-3],-1)
        logits = self.logits_net(logit_features)
        return Categorical(logits=logits)


    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class CNNJointActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, **kwargs,):
        super().__init__()

        obs_dim = observation_space.shape

        # #roll HWC input to CWH
        # self.feature_net = ImpalaCNN(tuple(np.roll(obs_dim,1)) ,**kwargs) # Roll input shape to CWH format
        self.feature_net = ImpalaCNN(obs_dim ,**kwargs)
        self.logits_net = nn.Linear(256, action_space.n)
        self.v_net = nn.Linear(256, 1)

    #Helper functions of distribution
    # def _distribution(self, obs):
    #     # expected image input is B x(??) x HWC, rearranging to (B??)CHW before forward and returning in B x ?? format
    #     obs_shape = obs.shape
    #     features = self.feature_net(obs.reshape(-1, *obs_shape[-3:]).transpose(-3, -1)).view(*obs_shape[:-3], -1)
    #     logits = self.logits_net(features)
    #     return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def features(self, obs):
        # expected image input is B x(??) x HWC, rearranging to (B??)CHW before forward and returning in B x ?? format
        obs_shape = obs.shape
        # features = self.feature_net(obs.reshape(-1, *obs_shape[-3:]).transpose(-3, -1)).view(*obs_shape[:-3], -1)
        features = self.feature_net(obs.reshape(-1, *obs_shape[-3:])).view(*obs_shape[:-3], -1)
        return features


    #Build v ,pi ,step functions, will build them as functions instead of standalone modules
    def v(self, obs):
        features = self.features(obs=obs)
        v = self.v_net(features).squeeze(-1) # Critical to ensure v has right shape.
        return v

    def pi(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        features = self.features(obs=obs)
        logits = self.logits_net(features)
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def forward(self, obs, act=None):
        features = self.features(obs=obs)
        logits = self.logits_net(features)
        pi = Categorical(logits=logits)
        v = self.v_net(features).squeeze(-1)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, v, logp_a

    def step(self, obs):
        with torch.no_grad():
            pi,v, _ = self(obs)
            a = pi.sample()
            logp_a = self._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), pi.logits.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class StateActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, **kwargs,):
        super().__init__()
        #Only for discrete action and observation spaces
        obs_dim = observation_space.n
        act_dim = action_space.n
        self.act_dim = act_dim
        self.v_matrix = torch.nn.Parameter(torch.Tensor(obs_dim))
        self.pi_logit_matrix = torch.nn.Parameter(torch.Tensor(obs_dim,act_dim))
        self.ones = torch.ones([1,act_dim])
        torch.nn.init.zeros_(self.v_matrix)
        torch.nn.init.uniform_(self.pi_logit_matrix,-1,1)


    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    #Build v ,pi ,step functions, will build them as functions instead of standalone modules
    def v(self, obs):
        obs_shape = obs.shape
        v = torch.gather(input = self.v_matrix, index=obs.view(-1).long(),dim=0 ).view([*obs_shape])
        return v

    def pi(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        logits = torch.gather(input = self.pi_logit_matrix, index=(self.ones*obs.reshape(-1)).long(),dim=0 ).view([*obs_shape,-1] )
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def forward(self, obs, act=None):
        obs_shape = obs.shape
        v = torch.gather(input = self.v_matrix, index=obs.reshape(-1).long(),dim=0 ).view([*obs_shape])
        logits = torch.gather(input = self.pi_logit_matrix, index=(self.ones*obs.reshape([-1,1])).long(),dim=0 ).reshape([*obs_shape,-1] )
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, v, logp_a

    def step(self, obs):
        with torch.no_grad():
            pi,v, _ = self(obs)
            a = pi.sample()
            logp_a = self._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), pi.logits.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

from .aws_rnn import RNNWDAbridgedModel
class RNNAbridgedActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,n_ens=1,n_layers=1, n_hid=256,grad_clip=0.25, **kwargs, ):
        super().__init__()
        self.obs_dim = observation_space.shape
        self.n_ens=n_ens
        self.n_act = action_space.n
        self.obs_feature_net = ImpalaCNN(self.obs_dim, **kwargs)
        self.n_hid=n_hid
        self.n_layers=n_layers

        self.ac_kwargs={'n_ens':n_ens, 'n_layers':n_layers, 'n_hid':n_hid,'grad_clip':grad_clip,**kwargs}


        if self.n_layers>0:
            self.belief_lstm = RNNWDAbridgedModel(rnn_type='LSTM',  nhid=n_hid, ninp=256 + self.n_act, nlayers=n_layers)
        self.bn_v = nn.BatchNorm1d(n_hid)
        self.bn_pi = nn.BatchNorm1d(n_hid)
        self.logits_net = nn.Linear(n_hid, self.n_ens *self.n_act)
        self.v_net = nn.Linear(n_hid, self.n_ens)
        if grad_clip>0 and self.n_layers>0:
            for p in self.belief_lstm.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -grad_clip, grad_clip))


    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def obs_features(self, obs):
        '''
        :param obs: T x B x C W H image tensor
        :return: features: T x B x n_feat observation features
        '''
        # expected image input is T x B x C x W x H, rearranging to (TB)xCxWxH before forward and returning in TxBxfeat format
        obs_shape = obs.shape
        features = self.obs_feature_net(obs.reshape(-1, *obs_shape[-3:])).view(*obs_shape[:-3], -1)
        return features

    def rnn_features(self,obs_feats, acts, hc, dones=None, breakpoints=None):
        '''
        :param obs_feats: T x B x feat tensor with observation features (o_{t})
        :param acts: : T x B x n_A tensor with one-hot encoding of a_{t-1}
        :param hc: ([n_layers] x B x [hidden_size], [n_layers] x B x [hidden_size]) tuple of tensors with initial hidden/cell states of RNN
        :param dones: T x B tensor to indicate start of new sequence
        :return: rnn_features: T x B x n_feat full belief-state features, hc_n: ([n_layers] x B x [hidden_size], [n_layers] x B x [hidden_size]) updated hiddens
        '''
        if self.n_layers>0:
            input_feats = torch.cat([obs_feats, acts],dim=-1)
            rnn_features, hc_n=self.belief_lstm(input_feats, hc, dones=dones, breakpoints=breakpoints)
        else:
            rnn_features, hc_n = obs_feats, hc

        return rnn_features, hc_n


    def forward(self, obs, prev_act, act=None,  hc=None, dones=None, breakpoints=None):#head_idx =None):
        '''
        :param obs: T x B x *obs_dim tensor with observations (o_{t})
        :param prev_act: T x B x n_act tensor with one-hot encoding of previous action (a_{t-1})
        :param act: T x B  tensor with index of current action (a_{t})
        # :param head_idx: T X B  tensor with index of active ensemble head
        :param hc: ([n_layers] x B x [hidden_size], [n_layers] x B x [hidden_size]) tuple of tensors with initial hidden/cell states of RNN
        :param dones: T x B tensor to indicate start of new sequence
        :param breakpoints: list of indices where some sequence in batch is reset (np.where(dones.max(0))
        :return: pi, v, logp_a, hc_n, {extras}
        '''
        # T,B = obs.shape[:2]
        obs_feats = self.obs_features(obs=obs)
        rnn_feats, hc_n = self.rnn_features(obs_feats=obs_feats,acts=prev_act, hc=hc, dones=dones, breakpoints=breakpoints)
        bn_v_feats = self.bn_v(rnn_feats.transpose(-1,-2)).transpose(-1,-2)
        bn_pi_feats = self.bn_pi(rnn_feats.transpose(-1,-2)).transpose(-1,-2)

        logits = self.logits_net(bn_pi_feats)
        logits = logits.reshape([*logits.shape[:-1], self.n_ens, self.n_act])
        v = self.v_net(bn_v_feats)
        pi = Categorical(logits=logits)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, v, logp_a,hc_n, {}

    def step(self, obs, prev_act, hc=None, average_ens=False):
        with torch.no_grad():
            pi,v, _, hc_n, extras = self(obs, prev_act, hc=hc)
            if average_ens:
                extras['full_probs'] = pi.probs.cpu().numpy()
                pi= Categorical(probs=pi.probs.mean(-2))
            a = pi.sample()
            logp_a = self._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), pi.logits.cpu().numpy(), (hc_n[0].cpu().numpy(), hc_n[1].cpu().numpy()), extras


    def act(self, obs):
        return self.step(obs)[0]
