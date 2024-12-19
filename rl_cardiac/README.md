# RLVNS
# Reinforcement Learning for Vagus Nerve Stimulation



This is a convenient gym environment for developing and comparing interaction of RL agents with several types of rat cardiac models, considering healthy/hypertension and with/without exercise. The TCN model is embeded into the original rat cardiac model as a reduced-order surrogate, which can speed up training.

Adapted for the course project. The RL algorithms are removed, and the TCN model is pre-trained.

Note: since I know sklearn really hate loading stuff from a different version, I am using the exact same version of sklearn and tensorflow as the original repo. If you want to use a different version, you can try to change the version to see if it works.

### Installation
See the readme file in the parent directory for installation instructions.

### Usage
We have 4 available rat types: 'healthy_stable', 'hypertension_stable', 'healthy_exercise', 'hypertension_exercise'. 
Get the TCN model with:
```
from tcn_model import TCN_config
tcn_model = TCN_config(rat_type)  # need to be a string from the 4 available rat types
```
Then the gym environment can be created with:
```
from cardiac_model import CardiacModel_Env
env = CardiacModel_Env(tcn_model, rat_type)
# noise level is set to 0 by default, should be changed to see if your agent can handle noise once it works well without noise
# env = CardiacModel_Env(tcn_model, rat_type, noise_level)  
```
  
