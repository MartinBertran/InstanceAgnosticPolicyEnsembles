import numpy as np
import torch
import iape.iape.backend.core as core
import torch.nn.utils.prune as prune

#Experience buffer
class RnnDeviantBufferLong:
    """
    A buffer for storing trajectories experienced by a Deviant agent interacting
    with the environment. in TB(?) format
    """
    def __init__(self, size, n_env , n_t, obs_dim, n_actions=1, n_minibatch=8,n_rnn=2, n_feat=256):
        #Length, n_envs (parallel tracks), rollout, n_minibatch for sampling
        # time*n_envs*whatever
        #effective size will be time

        self.act_buf = np.zeros((size, n_env,), dtype=np.float32)  # Actions
        self.prev_oh_act_buf = np.zeros((size, n_env , n_actions),dtype=np.float32)  # previous one_hot action
        self.rew_buf = np.zeros((size, n_env), dtype=np.float32)  # Reward buffer
        self.lvl_buf = np.zeros((size,n_env), dtype=np.int32) #lvl buffer
        self.logp_buf = np.zeros((size,n_env), dtype=np.float32) #logprob buffer of action taken
        self.done_buf = np.zeros((size,n_env), dtype=bool)  #Terminal state buffer
        self.obs_buf = np.zeros((size, n_env, *obs_dim),dtype=np.float32)  # Observations
        self.prev_h_buf = np.zeros((size, n_env, n_rnn, n_feat),dtype=np.float32)  # previous h
        self.prev_c_buf = np.zeros((size, n_env, n_rnn, n_feat),dtype=np.float32)  # previous c


        self.n_t , self.n_env = n_t, n_env
        self.n_minibatch = n_minibatch
        self.ptr, self.size, self.max_size = 0, size, 0

    def store(self, obs, act, rew, logp, done, prev_oh_act, h, c, lvl=None):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # assert self.ptr+self.n_env <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.prev_oh_act_buf[self.ptr] = prev_oh_act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.prev_h_buf[self.ptr] = np.swapaxes(h,0,1)
        self.prev_c_buf[self.ptr] = np.swapaxes(c,0,1)
        if lvl is not None:
            self.lvl_buf[self.ptr] = lvl

        # update pointer and filled capacity
        self.ptr = (self.ptr+1)%self.size
        self.max_size = np.minimum(self.max_size+1, self.size)

    def update_rnn_buf(self, h, c, indices):
        self.prev_h_buf[indices] = h
        self.prev_c_buf[indices] = c


    def get_idx(self, idx=None, lenght=None):
        """
        Get minibatch from data
        """
        if idx is None:
            idx = np.arange(self.max_ptr)

        # breakpoints=list(sorted(set(list(np.where(self.done_buf[idx:idx+lenght].sum(0)>0.5)[0])+[lenght]))) #Review carefully
        # breakpoints = list(sorted(set(list(np.where(self.done_buf[idx:idx + lenght].sum(1) > 0.5)[0]) + [lenght])))  # is any environment done in that timepoint?
        data = dict(obs=self.obs_buf[idx:idx+lenght], act=self.act_buf[idx:idx+lenght],
                    logp=self.logp_buf[idx:idx+lenght],dones=self.done_buf[idx:idx+lenght],
                    rew=self.rew_buf[idx:idx+lenght],
                    prev_oh_acts=self.prev_oh_act_buf[idx:idx+lenght],
                    prev_h_buf=self.prev_h_buf[idx:idx+lenght],
                    prev_c_buf = self.prev_c_buf[idx:idx+lenght],
                    )
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        # data['breakpoints']=torch.as_tensor(breakpoints, dtype=torch.long)
        data['lvl']=torch.as_tensor(self.lvl_buf[idx:idx+lenght], dtype=torch.long)
        return data

    def sample_minibatch(self,n_minibatch=None, device=None):
        n_minibatch = n_minibatch if n_minibatch else self.n_minibatch
        total_minibatch = 0
        data_list=[]
        #sample random (valid) starting times and fill data buffer, then concatenate along 'env' dimension and return data batch
        while total_minibatch<n_minibatch:
            start_time = np.random.randint(0, self.max_size-self.n_t)
            # if start_time <= self.ptr and start_time+self.n_t>self.ptr: #crosses memory index
            #     continue
            data_list.append(self.get_idx(start_time,self.n_t))
            total_minibatch += self.n_env

        #Concatenate, trim, and send to device
        mini_idx = np.arange(total_minibatch)
        np.random.shuffle(mini_idx)
        mini_idx = np.sort(mini_idx[:n_minibatch])
        data_mb ={}

        for key in data_list[0].keys():
            data_mb[key] = torch.cat([d[key] for d in data_list])[:,mini_idx]
        #REMAKE BREAKPOINTS!
        breakpoints = list(sorted(set(list(np.where(data_mb['dones'].sum(1) > 0.5)[0]) + [self.n_t-1])))
        data_mb['breakpoints'] = torch.as_tensor(breakpoints, dtype=torch.long)

        if device is not None:
            for key in data_list[0].keys():
                if key in ['rew','logp', ]:
                    continue
                data_mb[key] = data_mb[key].to(device)

        return data_mb

    def get_indexed_minibatch(self,start_t=0, start_e=0,lenght=256,n_minibatch=8, device=None):

        data = dict(obs=self.obs_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    act=self.act_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    logp=self.logp_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    dones=self.done_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    rew=self.rew_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    prev_oh_acts=self.prev_oh_act_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    prev_h_buf=self.prev_h_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    prev_c_buf = self.prev_c_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch],
                    )
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        breakpoints = list(sorted(set(list(np.where(data['dones'].sum(1) > 0.5)[0]) + [lenght-1])))  # is any environment done in that timepoint?
        data['breakpoints']=torch.as_tensor(breakpoints, dtype=torch.long)
        data['lvl']=torch.as_tensor(self.lvl_buf[start_t:start_t+lenght, start_e:start_e+n_minibatch], dtype=torch.long)

        if device is not None:
            for key in data.keys():
                if key in ['rew','logp', ]:
                    continue
                data[key] = data[key].to(device)
        return data

def get_ema(x, eta=0.7):
    if len(x)>1:
        x = np.array(x)
    else:
        x =np.zeros([1])
    for i in range(1,x.size):
        x[i] += eta*(x[i-1]-x[i])
    return x

def inputs_to_tensor(o_t, a_oh_tm1, h_tm1, c_tm1, device=None):
    if device is not None:
        tensor_o_t = torch.as_tensor(o_t, dtype=torch.float32).to(device)
        tensor_a_oh_tm1 = torch.as_tensor(a_oh_tm1, dtype=torch.float32).to(device)
        tensor_h_tm1 = torch.as_tensor(h_tm1, dtype=torch.float32).to(device)
        tensor_c_tm1 = torch.as_tensor(c_tm1, dtype=torch.float32).to(device)
    else:
        tensor_o_t = torch.as_tensor(o_t, dtype=torch.float32)
        tensor_a_oh_tm1 = torch.as_tensor(a_oh_tm1, dtype=torch.float32)
        tensor_h_tm1 = torch.as_tensor(h_tm1, dtype=torch.float32)
        tensor_c_tm1 = torch.as_tensor(c_tm1, dtype=torch.float32)
    return tensor_o_t, tensor_a_oh_tm1, tensor_h_tm1, tensor_c_tm1

def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False