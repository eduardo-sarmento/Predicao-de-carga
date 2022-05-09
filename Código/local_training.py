from tabnanny import verbose
import Smart_home
import sys
import tensorflow as tf
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Dict
import numpy as np
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn import svm


def NMSE_metric(y_true, y_pred):
    #nmse = K.mean(K.square(y_true - y_pred)/K.square(y_true), axis=-1)
    nmse = K.square(y_true - y_pred)/K.square(y_true + 0.0000000001)
    return nmse  


if __name__ == "__main__":
    path = sys.argv[1]
    df = Smart_home.load_data(path)
    #plt.plot(df.index)
    #plt.show()
    aux = path.split('/')
    ts_inputs = tf.keras.Input(shape=(len(df.columns)-1,1))
    
    #print(df.isnull().values.any())
    if 'RealPowerWatts_Circuit' in df.columns:
        Y = np.array(df['RealPowerWatts_Circuit'])
        X = np.array(df.drop('RealPowerWatts_Circuit', 1))
    else:
        Y = np.array(df['Watts'])
        X = np.array(df.drop('Watts', 1))        
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)
    col_mean = np.nanmean(x_train, axis=0)

    #Find indices that you need to replace
    inds = np.where(np.isnan(x_train))

    #Place column means in the indices. Align the arrays using take
    x_train[inds] = np.take(col_mean, inds[1])


    col_mean = np.nanmean(x_test, axis=0)

    #Find indices that you need to replace
    inds = np.where(np.isnan(x_test))

    #Place column means in the indices. Align the arrays using take
    x_test[inds] = np.take(col_mean, inds[1])

    #print(x_train)
    #print(np.exp(y_test))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    y_train = scaler.fit_transform(y_train.reshape(-1,1))
    x_test = scaler.fit_transform(x_test)
    y_test = target_scaler.fit_transform(y_test.reshape(-1,1))
    x_train = x_train[...,None]
    x_test = x_test[...,None]
    #print(y_test)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(y_train.shape[1]))
    model.add(LSTM(200, activation='sigmoid',return_sequences=True))
    
    model.add(TimeDistributed(Dense(100, activation='relu')))
    #model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(Dense(1)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-128
    #model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    ##model.add(Dense(128, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    ##model.add(LSTM(units=128))#, return_sequences=True))#(ts_inputs))
    #model.add(Dropout(0.2))
    #model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,clipnorm=1),
              loss='mse',#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),#loss='mae',#tf.keras.losses.MeanSquaredError(),
              metrics=tf.keras.metrics.RootMeanSquaredError())#metrics='mae')

    #print(x_train)
#
    ## units=10 -> The cell and hidden states will be of dimension 10.
    ##             The number of parameters that need to be trained = 4*units*(units+2)
    #x = LSTM(units=128,return_sequences=True)(ts_inputs)
    #x = Dropout(0.2)(x)
    #outputs = Dense(1, activation='linear')(x)
    #model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,clipnorm=1),
    #              loss=tf.keras.losses.MeanSquaredError(),
    #              metrics=['mse'])
    svr = svm.SVR(C=1.0,epsilon=0.1)
    model_svr = svr.fit(x_train[:,:,0], y_train)
    model.fit(x_train, y_train, epochs=20, validation_split=0.2, batch_size=128, steps_per_epoch=3)
    pred = model.predict(x_test)
    #print(pred.shape)
    #print(pred)
    loss, mse = model.evaluate(x_test, y_test)
    #pred = scaler.inverse_transform(pred)
    #y_test = scaler.inverse_transform(y_test)
    #print(y_test)
    pred_rescaled = target_scaler.inverse_transform(pred[:,:,0])
    y_test_rescaled =  target_scaler.inverse_transform(y_test)
    score = r2_score(y_test_rescaled, pred_rescaled)
    print('R-squared score for the test set CNN-LSTM:', round(score,4))

    pred = svr.predict(x_test[:,:,0])
    pred_rescaled = target_scaler.inverse_transform(pred)
    score = r2_score(y_test_rescaled, pred_rescaled)
    print('R-squared score for the test set MLP:', round(score,4))
    
    plt.plot(pred[:3600], label="Predicted")
    plt.plot(y_test_rescaled[:3600], label="Actual")
    plt.legend(loc="upper left")
    plt.xlabel('Time')
    plt.ylabel('Watts')
    plt.title("Predicted vs. Actual Power Consumption")
    plt.show()


#
    ## Plot all predictions
    plt.plot(pred_rescaled[:3600], label="Predicted")
    plt.plot(y_test_rescaled[:3600], label="Actual")
    plt.legend(loc="upper left")
    plt.xlabel('Time')
    plt.ylabel('Watts')
    plt.title("Predicted vs. Actual Power Consumption")
    plt.show()