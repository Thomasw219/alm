import collections

import numpy as np
import utils
import gym

def make_agent(env, device, cfg):
    
    if cfg.agent == 'alm':
        
        from agents.alm import AlmAgent

        num_states = np.prod(env.observation_space.shape)
        num_actions = np.prod(env.action_space.shape)
        action_low = env.action_space.low[0]
        action_high = env.action_space.high[0]

        if cfg.id == 'Humanoid-v2':
            cfg.env_buffer_size = 1000000
        buffer_size = min(cfg.env_buffer_size, cfg.num_train_steps)

        agent = AlmAgent(device, action_low, action_high, num_states, num_actions,
                            buffer_size, cfg.gamma, cfg.tau, cfg.target_update_interval,
                            cfg.lr, cfg.max_grad_norm, cfg.batch_size, cfg.seq_len, cfg.lambda_cost,
                            cfg.expl_start, cfg.expl_end, cfg.expl_duration, cfg.stddev_clip, 
                            cfg.latent_dims, cfg.hidden_dims, cfg.model_hidden_dims,
                            cfg.wandb_log, cfg.log_interval
                            )
                            
    else:
        raise NotImplementedError

    return agent

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_length: int):
        super().__init__(env)
        self._max_episode_length = max_episode_length
        self._current_episode_length = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action.copy())
        self._current_episode_length += 1

        if self._current_episode_length >= self._max_episode_length:
            done = True
            info['TimeLimit.truncated'] = True
        else:
            info['TimeLimit.truncated'] = False
            done = False

        return next_state, reward, done, info

    def reset(self, **kwargs):
        self._current_episode_length = 0
        return self.env.reset(**kwargs)

class DMObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = env.observation_space['observations']

    def observation(self,obs):
        return obs['observations']

class DoneZeroRewardWrapper(gym.Wrapper):
    '''
    Sets reward at any timestep after first termination event to 0
    (so you'll still get the reward associated with terminating, but will be 0 afterwards)
    Also acts as a no termination wrapper
    '''
    def __init__(self,env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        self.done = False
        return self.env.reset()
    
    def step(self,action):
        next_state,reward,d,info= self.env.step(action)
        if self.done:
            reward = 0
        
        # if the episode would normally terminate, update our own "done" flag
        if d:
            self.done = True

        return next_state,reward,False,info

def make_env(cfg):
    if cfg.benchmark == 'gym':
        import gym
        if cfg.id == 'AntTruncatedObs-v2' or cfg.id == 'HumanoidTruncatedObs-v2':
            utils.register_mbpo_environments()

        def get_env(cfg):
            env = gym.make(cfg.id, **({'environment_kwargs' : {'flat_observation':True}} if 'dm2gym' in cfg.id else {})) 
            if 'dm2gym' in cfg.id:
                env = DMObsWrapper(env)
            if 'Walker2d' in cfg.id:
                env = DoneZeroRewardWrapper(env)
            env = EpisodeLengthWrapper(env, 100)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed=cfg.seed)
            env.observation_space.seed(cfg.seed)
            env.action_space.seed(cfg.seed)
            return env 

        return get_env(cfg), get_env(cfg)
    
    else:
        raise NotImplementedError