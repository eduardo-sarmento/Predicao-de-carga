import glob
import warnings
from certifi import where
warnings.filterwarnings("ignore") 
import pandas as pd
from datetime import datetime
import math
import numpy as np
from numpy import loadtxt
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Activation,TimeDistributed, Input,RepeatVector,GlobalAveragePooling1D,BatchNormalization
import tensorflow_addons as tfa
from tensorflow_addons.rnn import LayerNormLSTMCell
from statsmodels.tsa import stattools
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from pickle import dump, load
from pandas_tfrecords import pd2tf, tf2pd


import re

np.random.seed(0)

def remove_lag(pred_rescaled,y_test_rescaled,x_test,features_scalers):
  lag_real = features_scalers['scaler_lag_Watts_1'].inverse_transform(x_test['lag_Watts_1'].values.reshape(-1, 1))
  pred_rescaled = np.add(pred_rescaled,lag_real)
  y_test_rescaled = np.add(y_test_rescaled,lag_real)
  return pred_rescaled,y_test_rescaled

def adiciona_lag(dataframe, n, horizon,diff):
  for column in dataframe.columns:
        for i in range(n):
                
                  dataframe['lag_'+ column + '_' + str(i+1)] = dataframe[column].shift(i+1)
                  dataframe.dropna(inplace=True)
                
                  if i != 0:
                    dataframe['lag_difference_' + column + '_' + str(i+1)] = dataframe['lag_' + column + '_' + str(i)] - dataframe['lag_'+ column + '_' + str(i+1)]#dataframe['lag_'+ column + '_' + str(i+1)].diff(1)
                  else:
                    dataframe['lag_difference_' + column + '_' + str(i+1)] = dataframe[column] - dataframe['lag_'+ column + '_' + str(i+1)]
                
        for i in range(horizon):
                  dataframe['horizon_' + column + '_' + str(i+1)] = dataframe[column].shift(-(i+1))
                  
                  dataframe['lag_difference_horizon_' + column + '_' + str(i+1)] = dataframe['horizon_' + column + '_' + str(i+1)].diff(1)


def prepara_dataset(dataframe, n_steps, horizon, k_features, casa,diff=True,select=False):

    dataframe.columns = pd.io.parsers.base_parser.ParserBase({'names':dataframe.columns, 'usecols':None})._maybe_dedup_names(dataframe.columns)
    if k_features != 0:
      adiciona_lag(dataframe,n_steps,horizon,diff)
      if diff:
        columns = ['lag_difference_Watts_1']+dataframe.columns[dataframe.columns.str.contains('lag_difference_horizon_')].to_list()
      else:
        columns = ['Watts']+dataframe.columns[dataframe.columns.str.contains('horizon_')].to_list()
      Y = dataframe[columns]
      X = dataframe.drop(columns, 1)  

      x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False, random_state=0)

    else:

      aux = dataframe.index
      df = dataframe['Watts']
      df = df.to_frame()
      df.index = aux
      
      adiciona_lag(df,n_steps,horizon,diff)

      if diff:
        columns = ['lag_difference_Watts_1']+dataframe.columns[dataframe.columns.str.contains('lag_difference_horizon_')].to_list()
      else:
        columns = ['Watts']+dataframe.columns[dataframe.columns.str.contains('horizon_')].to_list()
      Y = df[columns]
      X = df.drop(columns, 1)
      x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False, random_state=0)

     

    x_train = x_train.fillna(x_train.mean().fillna(0))
    x_test = x_test.fillna(x_test.mean().fillna(0))
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())

    plt.figure()

    plt.plot(y_test, label="Real")
    plt.legend(loc="upper left")
    plt.xlabel('Time')
    plt.ylabel('Watts')
    #plt.show()

    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    scalers={}
    for column in x_train.columns:
      scaler = MinMaxScaler(feature_range=(-1,1))
      s_s = scaler.fit_transform(x_train[column].values.reshape(-1, 1))
      s_s=np.reshape(s_s,len(s_s))
      scalers['scaler_'+ column] = scaler
      x_train[column]=s_s

    for column in x_test.columns:
      scaler = scalers['scaler_'+column] 
      s_s = scaler.transform(x_test[column].values.reshape(-1, 1))
      s_s=np.reshape(s_s,len(s_s))
      x_test[column]=s_s
    
    y_train = target_scaler.fit_transform(y_train)
    y_test = target_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test, target_scaler, scalers

def load_data(root_path):
  root = root_path
  data = []
  power_directories=['/home/nocs/TCC/Dataset/2013/homeA-all/homeA-circuit/','/home/nocs/TCC/Dataset/2013/homeB-all/homeB-power/','/home/nocs/TCC/Dataset/2013/homeC-all/homeC-power/']
  all_directories = glob.glob(root+'*/')
  all_directories = [x for x in all_directories if x in power_directories]
  for dir in all_directories:

      all_files = [f for f in  glob.glob(dir + "/*.csv")]
      print(dir)
      header = loadtxt(dir + "FORMAT",dtype=str,comments="#", delimiter=",", unpack=False)
      dfs = []
      dfs_p1 = []
      dfs_p2 = []
      for filename in all_files:
          df = pd.read_csv(filename, names=header)
          if 'TimestampUTC' in df.columns:
              df.set_index('TimestampUTC', inplace=True)
          else:
              df.set_index('TimestampLocal', inplace=True)
          df = df.loc[~df.index.duplicated(keep='first')]
          df.index = df.index.map(lambda x : datetime.utcfromtimestamp(x).strftime('%Y %m %d %H:%M:%S'))

          df.index = pd.to_datetime(df.index, format="%Y %m %d %H:%M:%S")

          if "p1" in filename:
            dfs_p1.append(df)
          elif "p2" in filename:
            dfs_p2.append(df)
          else: 
            dfs.append(df)
      if dfs:
        data.append(pd.concat(dfs))
      if dfs_p1:
        data.append(pd.concat(dfs_p1))
      if dfs_p2:
        data.append(pd.concat(dfs_p2))
  df_home = pd.concat(data,axis=1)

  df_home.sort_index(inplace=True)
  
  if 'Watts' in df_home.columns: 
    df_home['Watts'] = df_home['Watts'].replace('',np.nan).astype(float)
  else:
    df_home = df_home.rename(columns={'RealPowerWatts_Circuit': 'Watts'})
  
  df_home = df_home[df_home['Watts'].notna()]
  df_home = df_home.select_dtypes(include=np.number)
  df_resampled = df_home['Watts'].resample('15T').mean().to_frame()
  df_resampled['Standart_Deviation_Watts'] = df_home['Watts'].resample('15T').std()
  print(df_resampled['Watts'].autocorr(10))
  return df_resampled
      
def CNN_LSTM_compile(x_train, y_train, horizon):
    relu_initializer = tf.keras.initializers.HeUniform()
    tanh_initializer = tf.keras.initializers.GlorotNormal()
    x_train = np.array(x_train)[...,None]
    model = tf.keras.models.Sequential([
        Conv1D(64, kernel_size=3, padding='causal',strides=1,input_shape=(x_train.shape[1], x_train.shape[2])),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(strides=2),
        Conv1D(64, kernel_size=3,padding='causal', strides=1),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(strides=2),
        LSTM(64, return_sequences=False),#,recurrent_dropout=0.5),
        #BatchNormalization(),
        Activation('tanh'),
        Dense(32),
        #Dropout(0.20),
        Dense(horizon+1)
    ], name="lstm_cnn")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model

def model_builder(hp):
  hp_filters1 = hp.Int('f1', min_value=32, max_value=128, step=32)
  hp_filters2 = hp.Int('f2', min_value=32, max_value=128, step=32)
  hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
  hp_dense = hp.Int('dense', min_value=32, max_value=128, step=32)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  hp_activation = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])
  model = Sequential()
  
  model.add(Conv1D(hp_filters1, kernel_size=1, padding='causal', input_shape=(20, 1)))
  model.add(Activation('relu')),
  model.add(MaxPooling1D(strides=2)),
  
  model.add(Conv1D(filters=hp_filters2, kernel_size=1, activation='relu',padding='causal', use_bias=False,kernel_initializer='lecun_normal'))
  model.add(Activation('relu')),
  model.add(MaxPooling1D(strides=2)),

  model.add(LSTM(hp_units))
  model.add(Activation('tanh')),

  model.add(Dense(hp_dense))
  #model.add(Dropout(0.2))
  model.add(Dense(1,activation=hp_activation))
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate,clipnorm=1),
            loss='mse',
            metrics='mae')

  return model








