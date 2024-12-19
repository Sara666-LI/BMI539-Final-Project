
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from .utils_tcn import *
from gymnasium import spaces
from gymnasium.core import Env
print(tf.__version__)


def state_range(rat_type):
    if rat_type == 'healthy_stable':
        print('[cardiac_model/healthy_stable]')
        min_HR = -0.49
        max_HR = 0.51
        min_MAP = -0.36
        max_MAP = 0.64
       
    elif rat_type == 'healthy_exercise':
        print('[cardiac_model/healthy_exercise]')
        min_HR = -0.64
        max_HR = 0.36
        min_MAP = -0.62
        max_MAP = 0.38
   
    elif rat_type == 'hypertension_stable':
        print('[cardiac_model/hypertension_stable]')
        min_HR = -0.62
        max_HR = 0.38
        min_MAP = -0.67
        max_MAP = 0.33
     
    elif rat_type == 'hypertension_exercise':
        print('[cardiac_model/hypertension_exercise]')
        min_HR = -0.62
        max_HR = 0.38
        min_MAP = -0.67
        max_MAP = 0.33
        
    
    return min_HR, max_HR, min_MAP, max_MAP

class CardiacModel_Env(Env):
    def __init__(self, tcn_model, rat_type, noise_level=0.0):
        
        self.tcn_model = tcn_model
        # self.exp_name = exp_name
        self.action_space = spaces.Box(np.array([-1,-1,-1,-1,-1,-1]), np.array([1,1,1,1,1,1]))
        self.min_HR, self.max_HR, self.min_MAP, self.max_MAP = state_range(rat_type)
        self.noise_level = noise_level

        low = np.array([self.min_HR, self.min_MAP, self.min_HR, self.min_MAP], dtype=np.float32)
        high = np.array([self.max_HR, self.max_MAP, self.max_HR, self.max_MAP], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.sp_list = [np.array([0.074, -0.37]), np.array([-0.355, 0.108]), np.array([0.064, -0.16])]
        self.setpoints = self.sp_list[2] 
        self.initial_state = np.array([0.3, 0.25])
        self.total_reward = []

        self.consider_sp =  True
        self.ep_length = 100
        self.done = False
        self.current_step = 0
        self.current_reward = 0
        self.num_episode = 0
        self.listhistory = []
        self.num_history = 1
        self.seed = None
        self.previous_action = [[0]*6]
  

    def save_current_state(self, state):

        self.listhistory+=(list(state.flatten()))

        if (len(self.listhistory) /2) > self.num_history:
            self.listhistory.pop(0)
            self.listhistory.pop(0) 


    def reset_history(self, state):
        self.listhistory = list(state.flatten())

        while (len(self.listhistory) / 2) < self.num_history:
            self.listhistory+=(list(state.flatten()))


    def save_previous_action(self, action):
        self.previous_action.append(action)
        
    def step(self, u):

        u[0] = u[0] /2 + 0.28
        u[1] = u[1] /2 + 0.28
        u[2] = u[2] /2 + 0.28
        u[3] = u[3] /2 + 0.28
        u[4] = u[4] /2 + 0.28
        u[5] = u[5] /2 + 0.28
        in_TCN = np.concatenate((u, self.state), axis=0).reshape(1,1,8)
        self.state = self.tcn_model.predict(in_TCN)[0][0] + (np.random.normal(0, self.noise_level*0.5, np.shape(self.state)))
        self.save_current_state(self.state)

        rew = self.reward(self.state, self.setpoints, u)
        self.current_step += 1
        self.done = self.current_step >= self.ep_length
        self.current_reward += rew
        self.save_previous_action(u)
        
        if self.consider_sp:
            hist = (np.array(self.listhistory+ list(self.setpoints.flatten()))).reshape(1, -1).flatten()
            # we have no termination, always truncation at end of episode
            return hist, rew, False, self.done, {} # self.state
        else:
            return np.array(self.listhistory), rew, False, self.done, {} # self.state


    def reward(self, state, setpoint, action):
        return np.exp(-1*np.sum(np.power((state-setpoint),2)*5)) 

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        print('seed: ', self.seed)
        super().reset(seed=self.seed)
        self.num_episode += 1
        rand_sample = np.random.rand(2,2)
        self.state =  self.initial_state
        self.reset_history(self.state)
        self.total_reward.append(self.current_reward)
        dict = {'reward ': self.total_reward}
        df = pd.DataFrame(dict)
        self.current_step = 0
        self.current_reward = 0
        self.previous_action = [[0]*6]

        if self.consider_sp:

            hist = (np.array(self.listhistory+ list(self.setpoints.flatten()))).reshape(1, -1).flatten()
            return hist, []
        
        else:
            return np.array(self.listhistory), []

 
    


