import argparse
import numpy as np
import time
import tensorflow.compat.v1 as tf
from .utils_cardiac_TCN import rollout, policy
from .utils_tcn import *
import sys
import os

def TCN_config(rat_type):
    script_dir = os.path.dirname(__file__)
    print('script_dir', script_dir)
    if rat_type == 'healthy_stable':
        print('[tcn_model/healthy_stable]')
        # sys.path.append(os.path.join(script_dir,"Updated_Rat_Models_healthy_stable"))
        df_save_name = os.path.join(script_dir, "Healthy_500samples")

    elif rat_type == 'healthy_exercise':
        print('[tcn_model/healthy_exercise]')
        # sys.path.append(os.path.join(script_dir,"Updated_Rat_Models_healthy_exercise"))
        df_save_name = os.path.join(script_dir, "Healthy_WithExercise_500samples")

    elif rat_type == 'hypertension_stable':
        print('[tcn_model/hypertension_stable]')
        # sys.path.append(os.path.join(script_dir,"Updated_Rat_Models_hypertension_stable"))
        df_save_name = os.path.join(script_dir, "HyperTension_500samples")

    elif rat_type == 'hypertension_exercise':
        print('[tcn_model/hypertension_exercise]')
        # sys.path.append(os.path.join(script_dir,"Updated_Rat_Models_hypertension_exercise"))
        df_save_name = os.path.join(script_dir, "HyperTension_WithExercise_500samples")

    print('tcn_model', rat_type)
    input_width = 1
    loaded_models = dict()

    # we do not have other model types so the loop looks like this
    for model in ["TCN"]:
        saved_model_name = df_save_name + "/" + 'TCN' + "_" + str(input_width)
        print(f"Load model from {saved_model_name}")
        
        loaded_models[model] = tf.keras.models.load_model(saved_model_name)
    
    tcn_model = tf.keras.models.load_model(saved_model_name)

    return tcn_model
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rat_type', type=str, default='healthy_stable') 
    args = parser.parse_args()
    tcn_model = TCN_config(args.rat_type)
    print('[tcn_model/finish loading tcn model]')


