# %%
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Input, Dense, Activation, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import MeanSquaredError, MeanSquaredError

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
import os
from contextlib import redirect_stdout

import tensorflow as tf
import pandas as pd
tf.keras.backend.set_floatx('float32')

# %%
#features = 'paths-exp6-coords2'
features_list = ['paths-exp6-coords4',
                 'paths-exp4-coords2']

num_hidden_nodes = [16, 32]
activations = 2*['tanh']
num_atoms = 32
epoch = 900000
train = True
P_size = 3

f_train = pd.read_parquet('y_train.parquet').loc[:, [
    'fx', 'fy', 'fz']].astype('float32').values.reshape(-1, num_atoms, 3)

f_dev = pd.read_parquet('y_dev.parquet').loc[:, [
    'fx', 'fy', 'fz']].astype('float32').values.reshape(-1, num_atoms, 3)

f_test = pd.read_parquet('y_test.parquet').loc[:, [
    'fx', 'fy', 'fz']].astype('float32').values.reshape(-1, num_atoms, 3)


def create_model(num_atoms, L_size, P_size, num_hidden_layers, num_hidden_nodes):
    inputs = Input(shape=(num_atoms, L_size))
    #inputs = Input(shape=L_size)
    # Initialize the hidden layers
    hidden_layer = inputs
    for i in range(num_hidden_layers):
        hidden_layer = Dense(
            num_hidden_nodes[i], activation=activations[i])(hidden_layer)
    # Initialize the output layer with the predicted energy and forces for each atom
    E_i = Dense(P_size, activation='linear')(hidden_layer)
    # Create the model
    model = Model(inputs=inputs, outputs=E_i)
    return model


def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=0)  # Note the `axis=-1`


def schedule(epoch):
    return 1e-4


for features in features_list:
    path = os.getcwd()
    network = '_'.join([str(x) for x in num_hidden_nodes])
    num_hidden_layers = len(num_hidden_nodes)
    name = features+'--'+network+activations[0]

    f_type, n_hops, n_nei = features.split(
        '-')[0], *re.findall('\d+', features)

    #
    X_train = pd.read_parquet('X_'+f_type+'_train.parquet')
    X_dev = pd.read_parquet('X_'+f_type+'_dev.parquet')
    X_test = pd.read_parquet('X_'+f_type+'_test.parquet')
    #
    cols_hops = X_train.columns[X_train.columns.str.startswith(
        tuple(str(x)+'_(' for x in range(int(n_hops)+1)))]
    cols_neis = X_train.columns[X_train.columns.str.startswith(
        tuple(str(x)+'_[' for x in range(int(n_nei)+1)))]
    cols = cols_hops.to_list()+cols_neis.to_list()+['Atom']
    #
    X_train = X_train.loc[:, cols].copy()
    X_dev = X_dev.loc[:, cols].copy()
    X_test = X_test.loc[:, cols].copy()

    #

    #
    scaler = MinMaxScaler().fit(X_train)

    X_test = pd.DataFrame(scaler.transform(
        X_test), columns=X_test.columns, index=X_test.index)
    X_dev = pd.DataFrame(scaler.transform(
        X_dev), columns=X_dev.columns, index=X_dev.index)
    X_train = pd.DataFrame(scaler.transform(
        X_train), columns=X_train.columns, index=X_train.index)

    #
    X_train2 = X_train.to_numpy().reshape(-1, num_atoms, X_train.shape[-1])
    X_dev2 = X_dev.to_numpy().reshape(-1, num_atoms, X_dev.shape[-1])
    X_test2 = X_test.to_numpy().reshape(-1,  num_atoms, X_test.shape[-1])

    L_size = X_train2.shape[2]

    model = create_model(num_atoms, L_size, P_size,
                         num_hidden_layers, num_hidden_nodes)
    model.compile(optimizer='adam', loss=my_loss_fn, metrics='mae')

    model.summary()

    fpath = os.path.join(path+'/results', name)

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    with open(os.path.join(fpath, 'modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(
        fpath, 'checkpoints/'), monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False,
        mode='auto', save_frequency=1)
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(fpath, 'training.csv'), append=True)
    progbar_logger = tf.keras.callbacks.ProgbarLogger(
        count_mode="samples", stateful_metrics=None)

    if train:
        try:
            model = tf.keras.models.load_model(os.path.join(
                fpath, 'checkpoints'), custom_objects={'my_loss_fn': my_loss_fn})
            checkpoint.best = model.evaluate(X_dev2, f_dev, batch_size=16)[0]
        except:
            pass
        history = model.fit(X_train2, f_train,
                            batch_size=16,
                            epochs=epoch,
                            validation_data=(X_dev2, f_dev),
                            verbose=1, callbacks=[checkpoint, csv_logger, scheduler])

# %%
