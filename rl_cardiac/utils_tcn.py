#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March  23 12:26:26 2021

@author: zeydabadi
"""
# import libraries

import os
import sys
import time
import random
import itertools
import numpy as np
import pandas as pd

# import seaborn as sns
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

# Import tensorflow
import tensorflow as tf
#from kerashypetune import KerasGridSearch
print(f"Tensorflow version is: {tf.__version__}")

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, TimeDistributed, RepeatVector, Concatenate, Lambda, Reshape

import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras import backend as K


metrics = [
    tf.metrics.MeanAbsoluteError(),
    tf.metrics.MeanAbsolutePercentageError(),
    tf.metrics.MeanSquaredError()
]


def mat_to_df(path, filenames):

    frame = []
    for filename in filenames:
        mat_dict = loadmat(path + filename, squeeze_me=True)

        # drop the key/values that we don't want
        del mat_dict["__header__"]
        del mat_dict["__version__"]
        del mat_dict["__globals__"]

        # Convert mat_dict to mat_df
        mat_df = pd.DataFrame()

        for key in mat_dict.keys():
            mat_df[key] = mat_dict[key]

        frame.append(mat_df)

    return pd.concat(frame, ignore_index=True, axis=0)


# Safety for splitting the data between training, validation, and testing
def get_bounds(n, train_frac, valid_frac, cycle_length):

    a_frac = train_frac + valid_frac
    bound_1 = cycle_length * int(round((n * train_frac) / cycle_length))
    bound_2 = cycle_length * int(round((n * a_frac) / cycle_length))

    return bound_1, bound_2

# def inverse_get_bounds():



def split_data(df, cycle_length):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    print('n: ', n)

    # For splitting, we only pass the train fraction and the validation fraction
    train_end, valid_end = get_bounds(n, 0.7, 0.2, cycle_length)


    print('train_end: ', train_end)

    print('valid_end: ', valid_end)
    #train_df = df[0:int(n * 0.7)]
    #val_df = df[int(n * 0.7):int(n * 0.9)]
    #test_df = df[int(n * 0.9):]



    train_df = df[0:train_end]
    val_df = df[train_end:valid_end]
    test_df = df[valid_end:]

    print('len train_df: ', len(train_df))
    print('len val_df: ', len(val_df))
    print('len test_df: ', len(test_df))

    num_features = df.shape[1]

    return train_df, val_df, test_df, num_features, column_indices, train_end, valid_end

def normalize_data(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df, train_mean, train_std

def minmax_data_previous(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_max = train_df.max()
    train_min = train_df.min()

    #train_df = (train_df - train_mean) / (train_max - train_min) # should have been (train_df - train_min) / (train_max - train_min) to be [-0.5,0.5]
    #val_df = (val_df - train_mean) / (train_max - train_min)
    #test_df = (test_df - train_mean) / (train_max - train_min)

    train_df = 2*(((train_df - train_min) / (train_max - train_min))-0.5)
    val_df = 2*(((val_df - train_min) / (train_max - train_min))-0.5)
    test_df = 2*(((test_df - train_min) / (train_max - train_min))-0.5)

    return train_df, val_df, test_df, train_mean, train_max, train_min


def minmax_data(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_max = train_df.max()
    train_min = train_df.min()
    train_df = (train_df - train_mean) / (train_max - train_min)
    val_df = (val_df - train_mean) / (train_max - train_min)
    test_df = (test_df - train_mean) / (train_max - train_min)
    return train_df, val_df, test_df, train_mean, train_max, train_min



class WindowGenerator():
    def __init__(self,
                 input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 sequence_stride,
                 cycle_length,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.sequence_stride = sequence_stride
        self.cycle_length = cycle_length

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i
                for i, name in enumerate(label_columns)
            }
        self.column_indices = {
            name: i
            for i, name in enumerate(train_df.columns)
        }

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(
            self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(
            self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([
            labels[:, :, self.column_indices[name]]
            for name in self.label_columns
        ],
                          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=self.sequence_stride,
        shuffle=False,
        batch_size=32)
        #batch_size=(self.cycle_length - self.total_window_size + 1),
    

    ds = ds.map(self.split_window)# split_window = divides batch_of_sequences into
                                 # (input, label) pairs

    return ds

@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


def plot(self, model=None, plot_col=None, max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices,
                 inputs[n, :, plot_col_index],
                 label='Inputs',
                 marker='.',
                 zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices,
                    labels[n, :, label_col_index],
                    edgecolors='k',
                    label='Labels',
                    c='#2ca02c',
                    s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices,
                        predictions[n, :, label_col_index],
                        marker='X',
                        edgecolors='k',
                        label='Predictions',
                        c='#ff7f0e',
                        s=64)

        if n == 0:
            plt.legend()

    plt.xlabel("Cycles")


# Add methods to class
WindowGenerator.split_window = split_window
WindowGenerator.make_dataset = make_dataset
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
WindowGenerator.plot = plot
