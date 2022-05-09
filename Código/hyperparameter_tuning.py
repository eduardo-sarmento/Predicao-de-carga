from tabnanny import verbose
import Smart_home
import sys
import tensorflow as tf
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM 
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Dict
import numpy as np
from sklearn.model_selection import train_test_split
import keras_tuner as kt


def NMSE_metric(y_true, y_pred):
    #nmse = K.mean(K.square(y_true - y_pred)/K.square(y_true), axis=-1)
    nmse = K.square(y_true - y_pred)/K.square(y_true + 0.0000000001)
    return nmse  


if __name__ == "__main__":
    path = sys.argv[1]
    df = Smart_home.load_data(path)
    aux = path.split('/')
    ts_inputs = tf.keras.Input(shape=(len(df.columns)-1,1))
    
    #print(df.isnull().values.any())
    if 'RealPowerWatts_Circuit' in df.columns:
        Y = np.array(df['log_RealPowerWatts_Circuit'])
        X = np.array(df.drop('log_RealPowerWatts_Circuit', 1))
    else:
        Y = np.array(df['log_Watts'])
        X = np.array(df.drop('log_Watts', 1))        
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    x_train.fillna(df.mean().round(1), inplace=True)
    x_test.fillna(df.mean().round(1), inplace=True)
    #print(x_train)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    y_train = scaler.fit_transform(y_train.reshape(-1,1))
    x_test = scaler.fit_transform(x_test)
    y_test = scaler.fit_transform(y_test.reshape(-1,1))

    print(x_train)
    tuner = kt.Hyperband(Smart_home.model_builder,
                     objective='val_mse',
                     max_epochs=5,
                     factor=3,
                     directory='/home/nocs/TCC/' + aux[-2],
                     project_name='Tuning')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[stop_early],verbose=0)

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)
    hypermodel = tuner.hypermodel.build(best_hps)
    history = model.fit(x=x_train, y=y_train,  epochs=5, validation_split=0.2,verbose=0)
    print(history.history)
    val_acc_per_epoch = history.history['val_mse']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # Retrain the model
    hypermodel.fit(x=x_train, y=y_train, epochs=best_epoch, validation_split=0.2)
    pred = model.predict(x_test)
    eval_result = hypermodel.evaluate(x_test, y_test)
    print("[test loss, test accuracy]:", eval_result)
    #pred = pred[:, :, 0]
    #pred = scaler.inverse_transform(pred)
#
##
    ### Plot all predictions
    plt.plot(pred)
    plt.plot(Y)
    plt.xlabel('Time')
    plt.ylabel('Watts')
    plt.title("Predicted vs. Actual Power Consumption")
    plt.show()