from gym.envs.registration import register

register(
    id='oscillator-v0',
    entry_point='rl_dbs.gym_oscillator.envs:oscillatorEnv',
)
