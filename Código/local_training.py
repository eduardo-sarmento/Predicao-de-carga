from tabnanny import verbose
import pandas as pd
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
import random



if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    path = sys.argv[1]
    aux = path.split('/')
    casa = aux[-2][4]
    df = Smart_home.load_data(path)
    diff = False
    n_steps = 10
    horizon =  0 
    k_features = [0]
    es = EarlyStopping(monitor='val_loss', patience=20)
    for k in k_features:
        checkpoint = ModelCheckpoint('best_model'+ casa + '_' + str(k) + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        x_train,x_test, y_train, y_test, target_scaler, features_scalers = Smart_home.prepara_dataset(df,n_steps, horizon, k,casa,diff=diff)
        model = Smart_home.CNN_LSTM_compile(x_train, y_train, horizon)
        history = model.fit(x_train, y_train, epochs=100, validation_split=0.10, batch_size=32,verbose=0,callbacks=[es,checkpoint])
        model = load_model('best_model'+ casa + '_' + str(k) + '.h5')
        pred = model.predict(x_test)
        pred_rescaled = target_scaler.inverse_transform(pred)
        y_test_rescaled =  target_scaler.inverse_transform(y_test)
        if diff:
            pred_rescaled,y_test_rescaled = Smart_home.remove_lag(pred_rescaled,y_test_rescaled,x_test,features_scalers)

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Learning curve para ' + str(k) +' features para casa ' + casa )
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('Learning curve para ' + str(k) +' features para casa ' + casa +'.png') 
        plt.figure()
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('MAE curve para ' + str(k) +' features para casa ' + casa )
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('MAE curve para ' + str(k) +' features para casa ' + casa +'.png')  
     
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
        rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
        score = r2_score(y_test_rescaled, pred_rescaled)

        print('R-squared score para o conjunto de testes usando CNN-LSTM ' + str(k) +' features:', round(score,4))  
        print('MAE CNN-LSTM ' + str(k) +' features:', mae)
        print('RMSE CNN-LSTM ' + str(k) +' features:', rmse)

plt.figure()
plt.plot(pred_rescaled, label="Predito")
plt.plot(y_test_rescaled, label="Real")
plt.legend(loc="upper left")
plt.xlabel('Time')
plt.ylabel('Watts')
plt.title('Preditito vs. Real Consumo Eletrico CNN-LSTM ' + str(k) +' features para casa ' + casa)
plt.savefig('Preditito vs. Real Consumo Eletrico CNN-LSTM ' + str(k) +' features para casa ' + casa +'.png')  

    