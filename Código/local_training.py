from tabnanny import verbose

import Smart_home
import sys
import tensorflow as tf
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,GlobalAveragePooling1D,BatchNormalization,AlphaDropout,Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from matplotlib import pyplot as plt
from typing import Dict
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def CNN_LSTM_compile(x_train, x_test, y_train, y_test, target_scaler, horizon, casa):

  
    model = tf.keras.models.Sequential([
        Conv1D(64, kernel_size=3,activation='relu', padding='causal', use_bias=False,kernel_initializer='lecun_normal', input_shape=(x_train.shape[1], x_train.shape[2])),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', use_bias=False,kernel_initializer='lecun_normal'),
        
        BatchNormalization(scale=False),
        #Activation('selu'),
        #AlphaDropout(0.1),
        GlobalAveragePooling1D(),
        #MaxPooling1D(pool_size=2),
        #LSTM(128, activation='tanh', return_sequences=True),
        RepeatVector(y_train.shape[1]),
        LSTM(64, activation='tanh', return_sequences=False, recurrent_dropout=0.4),
        Dense(32),
        Dropout(0.2),
        Dense(horizon+1),
        #Activation('linear', dtype='float32')
    ], name="lstm_cnn")
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
    #optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    #model = Sequential()
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(RepeatVector(y_train.shape[1]))
    #model.add(LSTM(units=64, activation='tanh'))
    #model.add(Dropout(0.2))
    #model.add(Dense(32))
    #model.add(TimeDistributed(Dropout(0.2)))
    #model.add(TimeDistributed(Dense(horizon+1)))
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,clipnorm=1),
    #          loss='mse',#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),#,#loss='mae',#tf.keras.losses.MeanSquaredError(),
    #          metrics='mae')

    return model


if __name__ == "__main__":
    np.random.seed(0)
    path = sys.argv[1]
    aux = path.split('/')
    casa = aux[-2][4]
    #mixed_precision.set_global_policy('mixed_float16')
    df = Smart_home.load_data(path)

    n_steps = 10
    horizon =  0  
    k_features = [0,1,2,4,8]
    checkpoint = ModelCheckpoint('best_model'+ casa + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=10)
    for k in k_features:
        
        x_train, x_test, y_train, y_test, target_scaler = Smart_home.prepara_dataset(df,n_steps, horizon, k)
    #
        x_train = x_train[...,None]
        x_test = x_test[...,None]
#
        model = CNN_LSTM_compile(x_train, x_test, y_train, y_test, target_scaler, horizon, casa)
        history = model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=512,verbose=0,callbacks=[es,checkpoint])
        model = load_model('best_model'+ casa + '.h5')
        pred = model.predict(x_test)
        pred_rescaled = target_scaler.inverse_transform(pred)
#
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Learning curve para ' + str(k) +' features para casa ' + casa )
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('Learning curve para ' + str(k) +' features para casa ' + casa +'.png')  
#
        y_test_rescaled =  target_scaler.inverse_transform(y_test)
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
        mape = mean_absolute_percentage_error(y_test_rescaled, pred_rescaled)
        mse = mean_squared_error(y_test_rescaled, pred_rescaled)
        rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
        score = r2_score(y_test_rescaled, pred_rescaled)
        print('R-squared score para o conjunto de testes usando CNN-LSTM ' + str(k) +' features:', round(score,4))
#
        print('MAE CNN-LSTM ' + str(k) +' features:', mae)
        print('MAPE CNN-LSTM ' + str(k) +' features:', mape)
        print('MSE CNN-LSTM ' + str(k) +' features:', mse)
        print('RMSE CNN-LSTM ' + str(k) +' features:', rmse)
#
        #   ## Plot all predictions
        plt.figure()
        plt.plot(pred_rescaled, label="Predito")
        plt.plot(y_test_rescaled, label="Real")
        plt.legend(loc="upper left")
        plt.xlabel('Time')
        plt.ylabel('Watts')
        plt.title('Preditito vs. Real Consumo Eletrico CNN-LSTM ' + str(k) +' features para casa ' + casa)
        ##plt.show()
        plt.savefig('Preditito vs. Real Consumo Eletrico CNN-LSTM ' + str(k) +' features para casa ' + casa +'.png')  

    