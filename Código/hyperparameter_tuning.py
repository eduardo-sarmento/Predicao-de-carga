from tabnanny import verbose
import Smart_home
import sys
import tensorflow as tf
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Dict
import numpy as np
from sklearn.model_selection import train_test_split
import keras_tuner as kt


if __name__ == "__main__":
    path = sys.argv[1]
    aux = path.split('/')

    n_steps = 5
    horizon =  0  
    k_features = 0

    x_train, x_test, y_train, y_test, target_scaler= Smart_home.load_data(path, n_steps, horizon, k_features)

    x_train = x_train[...,None]
    x_test = x_test[...,None]
    
    tuner = kt.Hyperband(Smart_home.model_builder,
                     objective='val_mae',
                     max_epochs=1,
                     factor=3,
                     directory='/home/nocs/TCC/' + aux[-2],
                     #overwrite=True,
                     project_name='Tuning')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_train, y_train, epochs=1, validation_split=0.2, callbacks=[stop_early],verbose=1)
    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)
    print(f"Melhor numero de filtros primeira camada convolucional:{best_hps.get('f1')}")
    print(f"Melhor numero de filtros segunda camada convolucional:{best_hps.get('f2')}")
    print(f"Melhor numero de unidades da camada LSTM:{best_hps.get('units')}")
    print(f"Melhor learning rate:{best_hps.get('learning_rate')}")

    model = tuner.hypermodel.build(best_hps)
    hypermodel = tuner.hypermodel.build(best_hps)
    history = model.fit(x=x_train, y=y_train,  epochs=300, validation_split=0.2,verbose=0)
    print(history.history)
    val_acc_per_epoch = history.history['val_mse']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Melhor epoch: %d' % (best_epoch,))

    # Retrain the model
    hypermodel.fit(x=x_train, y=y_train, epochs=best_epoch, validation_split=0.2)
    pred = model.predict(x_test)
    eval_result = hypermodel.evaluate(x_test, y_test)
    print("[teste loss, teste MAE]:", eval_result)