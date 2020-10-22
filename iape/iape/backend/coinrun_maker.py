from collections import deque
import gym
from gym import spaces
from gym.spaces import Box
from gym import ObservationWrapper

from iape.utils.vec_env import DummyVecEnv




# from gym.wrappers import GrayScaleObservation, FrameStack
import numpy as np
def make_coin_env(num_env, game_type='standard', set_seed=-1, num_levels=500, **kwargs):
    #game_type in ''standard', 'platform', 'maze'
    from coinrun import coinrunenv, wrappers
    from coinrun.setup_utils import setup_and_load
    setup_args =  setup_and_load(False, game_type=game_type, set_seed=set_seed, num_levels=num_levels, **kwargs)
    print('SETUP ARGS')
    print(setup_args)
    env = coinrunenv.make(game_type, num_env)
    env = wrappers.add_final_wrappers(env)
    env = ToPytorchObservation(env)
    return env

def make_procgen_env(num_env,start_level=0, game_type ='procgen:procgen-coinrun-v0', num_levels=500,
                  paint_vel_info=False,use_data_augmentation=False, restrict_themes=False, native=False, difficulty=0,**kwargs):
    start_level = int(np.maximum(0,start_level))
    env_name = game_type.split('procgen:procgen-')[1].split('-v0')[0]
    dist_mode = 'hard' if difficulty ==1 else 'easy'

    import procgen
    from procgen import ProcgenEnv
    if native: #custom procgen, can perform CUTOUT in environment natively
        env = ProcgenEnv(
            num_envs=num_env, env_name=env_name, distribution_mode=dist_mode,
            num_levels=num_levels, start_level=start_level,
            paint_vel_info=paint_vel_info,
            use_data_augmentation=use_data_augmentation,
            restrict_themes=restrict_themes,
            rand_seed = 0,
            **kwargs
        )
        env.reward_range = (0, 10)
        env.metadata = {}
        env = ToPytorchProcgenObservation(env)
    else:
        env = ProcgenEnv(
            num_envs=num_env, env_name=env_name, distribution_mode=dist_mode,
            num_levels=num_levels, start_level=start_level,
            paint_vel_info=paint_vel_info,
            restrict_themes=restrict_themes,
            rand_seed=0,
            **kwargs
        )
        env.reward_range = (0, 10)
        env.metadata = {}
        env = ToPytorchCutoutObservation(env,use_cutout=use_data_augmentation)
    return env

def make_atari_env(num_env, env_name):

    env = DummyVecEnv([lambda : gym.make(env_name) for _ in range(num_env)])
    env.reward_range=[0,1]
    env = ResizeObservation(env=env, shape=64)
    env = ToPytorchObservation(env)
    return env



class ToPytorchObservation(ObservationWrapper):
        def __init__(self, env):
            super(ToPytorchObservation, self).__init__(env)
            self.f = lambda obs: np.moveaxis(obs,[-3,-2,-1], [-2,-1,-3]).astype('float32')/255.0

            previous_shape = self.observation_space.shape
            previous_low = np.min(self.observation_space.low)
            previous_high = np.max(self.observation_space.high)
            new_shape = (*previous_shape[:-3],previous_shape[-1],previous_shape[-3],previous_shape[-2])
            self.observation_space = Box(low=previous_low/255.0, high=previous_high/255.0, shape=new_shape,
                                                dtype=np.float32)

        def observation(self, observation):
            processed_obs = self.f(observation)
            return processed_obs

class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)


        previous_shape = self.observation_space.shape
        previous_low = np.min(self.observation_space.low)
        previous_high = np.max(self.observation_space.high)
        new_shape = (*previous_shape[:-3], shape,shape, previous_shape[-1])
        self.shape = new_shape
        self.observation_space = Box(low=previous_low, high=previous_high , shape=new_shape,
                                     dtype=np.float32)
        import cv2
        self.f =  lambda observation: np.concatenate([cv2.resize(obs, (shape,shape))[np.newaxis,...] for obs in observation])

    def observation(self, observation):
        processed_obs = self.f(observation)
        return processed_obs

class CutoutObservation(ObservationWrapper):
    def __init__(self, env):
        super(CutoutObservation, self).__init__(env)

        self.width, self.height= self.observation_space.shape[0:2]
        self.max_rand_dim = .35
        self.min_rand_dim = .1
        self.max_blobs=6
        self.num_envs = self.env.num_envs
        self.x_dim_min = np.floor(self.min_rand_dim*self.width)
        self.x_dim_max = np.floor(self.max_rand_dim*self.width)
        self.y_dim_min = np.floor(self.min_rand_dim*self.height)
        self.y_dim_max = np.floor(self.max_rand_dim*self.height)


    def observation(self, observation):
        #batched range sampling
        x_mins = np.random.randint(self.width, size=[self.num_envs,self.max_blobs])
        x_maxs = np.minimum(x_mins + np.random.randint(self.x_dim_min, self.x_dim_max,size=[self.num_envs,self.max_blobs]), self.width).astype('int')
        y_mins = np.random.randint(self.width, size=[self.num_envs,self.max_blobs])
        y_maxs = np.minimum(y_mins + np.random.randint(self.y_dim_min, self.y_dim_max,size=[self.num_envs,self.max_blobs]), self.height).astype('int')
        for env in range(self.num_envs):
            for blotch in range(np.random.randint(0,self.max_blobs+1)):
                # x_min, y_min = np.random.randint(self.width), np.random.randint(self.height)
                # x_max = np.minimum(x_min + np.random.randint(self.x_dim_min,self.x_dim_max),self.width).astype('int')
                # y_max = np.minimum(y_min + np.random.randint(self.y_dim_min,self.y_dim_max),self.height).astype('int')
                # observation[env,x_min:x_max,y_min:y_max,:] = np.random.randint(0,256,size=[1,1,1,3])
                observation[env,x_mins[env,blotch]:x_maxs[env,blotch],y_mins[env,blotch]:y_maxs[env,blotch],:] = np.random.randint(0,256,size=[1,1,1,3])
        return observation

class ToPytorchCutoutObservation(ObservationWrapper):
    def __init__(self, env, use_cutout=True):
        super(ToPytorchCutoutObservation, self).__init__(env)
        self.f = lambda obs: np.moveaxis(obs, [-3, -2, -1], [-2, -1, -3]).astype('float32') / 255.0

        previous_shape = self.observation_space['rgb'].shape
        previous_low = np.min(self.observation_space['rgb'].low)
        previous_high = np.max(self.observation_space['rgb'].high)
        self.new_shape = (*previous_shape[:-3], previous_shape[-1], previous_shape[-3], previous_shape[-2])
        self.observation_space = Box(low=previous_low / 255.0, high=previous_high / 255.0, shape=self.new_shape,
                                     dtype=np.float32)

        self.use_cutout=use_cutout
        self.width, self.height=previous_shape[0:2]
        self.max_rand_dim = .35
        self.min_rand_dim = .1
        self.max_blobs=6
        self.num_envs = self.env.num_envs
        self.x_dim_min = np.floor(self.min_rand_dim*self.width)
        self.x_dim_max = np.floor(self.max_rand_dim*self.width)
        self.y_dim_min = np.floor(self.min_rand_dim*self.height)
        self.y_dim_max = np.floor(self.max_rand_dim*self.height)


    def observation(self, observation):
        observation = observation['rgb']
        if self.use_cutout:
            observation = self.cutout(observation)
        observation = self.f(observation)
        return observation

    def cutout(self,observation):
        #batched range sampling
        x_mins = np.random.randint(self.width, size=[self.num_envs,self.max_blobs])
        x_maxs = np.minimum(x_mins + np.random.randint(self.x_dim_min, self.x_dim_max,size=[self.num_envs,self.max_blobs]), self.width).astype('int')
        y_mins = np.random.randint(self.width, size=[self.num_envs,self.max_blobs])
        y_maxs = np.minimum(y_mins + np.random.randint(self.y_dim_min, self.y_dim_max,size=[self.num_envs,self.max_blobs]), self.height).astype('int')
        for env in range(self.num_envs):
            for blotch in range(np.random.randint(0,self.max_blobs+1)):
                observation[env,x_mins[env,blotch]:x_maxs[env,blotch],y_mins[env,blotch]:y_maxs[env,blotch],:] = np.random.randint(0,256,size=[1,1,1,3])
        return observation

class ToPytorchProcgenObservation(ObservationWrapper):
    def __init__(self, env):
        super(ToPytorchProcgenObservation, self).__init__(env)
        self.f = lambda obs: np.moveaxis(obs, [-3, -2, -1], [-2, -1, -3]).astype('float32') / 255.0

        previous_shape = self.observation_space['rgb'].shape
        previous_low = np.min(self.observation_space['rgb'].low)
        previous_high = np.max(self.observation_space['rgb'].high)
        self.new_shape = (*previous_shape[:-3], previous_shape[-1], previous_shape[-3], previous_shape[-2])
        self.observation_space = Box(low=previous_low / 255.0, high=previous_high / 255.0, shape=self.new_shape,
                                     dtype=np.float32)


    def observation(self, observation):
        return self.f(observation['rgb'])
