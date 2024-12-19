import argparse
import numpy as np
from gymnasium import spaces
from gymnasium.core import Env
from tvb.simulator.lab import *
from tvb.simulator.plot.phase_plane_interactive import PhasePlaneInteractive


class TVBWrapper(Env):
    def __init__(self, timestep=10, history_len=2000, max_len=6000, dt=0.05):
        self.timestep = timestep  # the interaction cycle time between the agent and the environment, unit is ms
        self.history_len = history_len  # the length of history data, unit is ms - EEG sample rate is 1kHz so 2000ms is also 2000 samples
        # EEG data is normalized to [-1,1]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.history_len, 65), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.max_len = max_len  # the maximum length of the simulation, unit is timestep



        # model parameters adapted from https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/tutorials/tutorial_s6_ModelingEpilepsy.ipynb
        # notice: self.model is just the epileptor model, not the simulator
        self.model = models.Epileptor()
        self.model.variables_of_interest = ('x1', 'y1', 'z', 'x2', 'y2', 'g')
        self.model.x0 = np.ones((76))*-2.4
        # we hard code the regions for now and do not pass them as arguments
        # 62: rPHC, 47: rHC, 40: rAMYG, major epileptogenic regions
        self.model.x0[[62,47,40]] = np.ones((3))*-1.6
        # 69: rTCI, 72: rTCV, minor epileptogenic regions
        self.model.x0[[69,72]] = np.ones((2))*-1.8
        # in one single tutorial it is connectiity.Connectivity(load_default=True), in others it is connectivity.Connectivity.from_file(). And load_default is not a parameter of Connectivity. Ew
        self.conn = connectivity.Connectivity.from_file()
        # what does difference coupling functions mean?
        # https://github.com/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/coupling.py
        self.coupl = coupling.Difference(a=np.array([1.]))
        self.hiss = noise.Additive(nsig=np.array([0., 0., 0., 0.0003, 0.0003, 0.]))
        self.integrator = integrators.HeunStochastic(dt=dt, noise=self.hiss)
        self.dt = dt  # the dt for integrator, unit is ms
        # load the default region mapping
        rm = region_mapping.RegionMapping.from_file()

        # Initialise EEG Monitor with period in physical time
        mon_raw = monitors.Raw()
        mon_EEG = monitors.EEG.from_file()
        self.EEG_labels = mon_EEG.sensors.labels  # channel names
        mon_EEG.region_mapping=rm
        mon_EEG.period=1. 

        # Bundle them
        self.what_to_watch = (mon_raw, mon_EEG)
        
        self.simulator = simulator.Simulator(model=self.model, connectivity=self.conn,
                                coupling=self.coupl, 
                                integrator=self.integrator, monitors=self.what_to_watch)
        self.simulator.configure()
        weighting = np.zeros((76))
        # weighting[[69, 72]] = np.array([2.])
        self.weighting = weighting
        self.eqn_t = equations.Linear()
        self.eqn_t.parameters['a'] = 0.0

        # we need to run a bit as init condition
        # https://github.com/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/simulator.py#L169
        # tvb requires init condition in shape of t, len(model.state_variables), n_regions, model.number_of_modes
        # https://github.com/the-virtual-brain/tvb-root/blob/ce135636cd615dd8f3ab8cab1520fec9df06d6ac/tvb_library/tvb/datatypes/connectivity.py#L275 
        # t is ideally larger than horizon, which is max(delay/ dt) + 1, where delay is defined in connectivity as tract length/ speed
        # a signal taking 2 seconds to travel in brain sounds enough right? 
        # we run init for 4000ms - also since the example plot looks noisy in the first 2000ms, and we give 2000ms history to the agent
        # https://github.com/the-virtual-brain/tvb-root/blob/ce135636cd615dd8f3ab8cab1520fec9df06d6ac/tvb_library/tvb/simulator/models/base.py#L50
        # number of modes is just 1 by default and unchanged by the Epileptor model
        # https://github.com/the-virtual-brain/tvb-library/blob/db1d399951c9a71fcb70b7453a9659724862c4c5/tvb/simulator/models/epileptor.py#L285 
        # the state variables are ('x1', 'y1', 'z', 'x2', 'y2', 'g'). 
        # We need to get this internal states in addition to the projected observables
        self.init_length = max(4000,self.history_len*2)
        
        




    def reset(self,seed=None):
        self.nstep = 0
        self.integrator = integrators.HeunStochastic(dt=self.dt, noise=self.hiss)
        self.simulator = simulator.Simulator(model=self.model, connectivity=self.conn, integrator = self.integrator,
                                coupling=self.coupl, monitors=self.what_to_watch)
        self.simulator.configure()
        (traw, raw), (tEEG, EEG) = self.simulator.run(simulation_length=self.init_length)
        self._raw_history = raw
        eeg_nstep = EEG.shape[0]
        timestep_nstep = eeg_nstep / self.init_length * self.timestep
        # since we needed all raw data and changed voi, we calculate eeg back
        EEG = EEG[:,3,:,0] - EEG[:,0,:,0]
        EEG = EEG[-self.history_len:,:]
        self.EEG = EEG

        self.state = self.EEG
        # normalize only for state but not raw eeg
        # self.state /= (np.max(self.state,0) - np.min(self.state,0))
        # self.state -= np.mean(self.state,0)

        return self.state, []

    def step(self, action):
        self.nstep += 1
        self.eqn_t.parameters['b'] = 10
        weighting = self.weighting
        weighting[69] = action[0]
        weighting[72] = action[1]
        stimulus = patterns.StimuliRegion(temporal = self.eqn_t, connectivity=self.conn, weight=weighting)
        stimulus.configure_space()
        # handling raw history as init condition
        self.simulator = simulator.Simulator(model=self.model, connectivity=self.conn,
                                coupling=self.coupl, monitors=self.what_to_watch, integrator = self.integrator,
                                initial_conditions=self._raw_history, stimulus=stimulus)
        #TODO: stimulus from action
        self.simulator.configure()
        (traw, raw), (tEEG, EEG) = self.simulator.run(simulation_length=self.timestep)
        raw_nstep = raw.shape[0]
        # update raw history buffer
        self._raw_history = np.concatenate((self._raw_history[raw_nstep:,:,:,:], raw), axis=0)
        # update state
        new_EEG = (EEG[:,3,:,0] - EEG[:,0,:,0])
        self.EEG = np.concatenate((self.EEG[self.timestep:,:], new_EEG), axis=0)

        self.state = self.EEG
        # normalize only for state but not raw eeg
        # self.state /= (np.max(self.state,0) - np.min(self.state,0))
        # self.state -= np.mean(self.state,0)

        reward = self.reward(self.EEG, action)
        truncated = self.nstep >= self.max_len
        return self.state, reward, False, truncated, {}
    
    def reward(self, state, action):
        # reward function

        EEG_norm = np.average(np.sum((state - np.average(state,axis=0)) ** 2)/self.history_len)

        return - (EEG_norm)/100000000 + 5
    def render(self):
        pass