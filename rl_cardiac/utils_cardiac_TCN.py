import numpy as np
from gym import make

def rollout(env, pilco, timesteps, verbose=True, random=False, SUBS=1, render=False):
        X = []; Y = []; 
        #x, rew = env.reset()
        x = env.state
        ep_return_full = 0
        ep_return_sampled = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, np.array([x]), random) 
            u = np.array(np.array(list(u)).reshape(1,6)[0])
            print('--action --', u)
            for i in range(SUBS): 
                x_new, r = env.step(u)

                ep_return_full += r
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            Y.append([x_new - x])
            ep_return_sampled += r
            x = x_new

        return np.stack(X), np.stack(Y).reshape(-1,2), ep_return_sampled, ep_return_full


def policy(env, pilco, x, random):

    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, -1])[0, :]


class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()
