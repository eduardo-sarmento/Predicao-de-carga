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
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
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

def adiciona_lag(dataframe, n, horizon):
  for column in dataframe.columns:
    #Aplicando logaritmo na potencia real
       # dataframe['log_' + column] = np.log(dataframe[column],where=dataframe[column] > 0) 

        #Loop que calcula os valores da diferença entre a portencia atual, a potencia anterior e a potencia anterior da anterior
        for i in range(n):
                
                  dataframe[column + '_lag_' + str(i+1)] = dataframe[column].shift(i+1)

                  #dataframe['log_' + column + '_lag_' + str(i+1)] = np.log(dataframe[column + '_lag_' + str(i+1)],where=dataframe[column + '_lag_' + str(i+1)] > 0)
                  
                  dataframe['lag_difference_' + column + '_' + str(i+1)] = abs(dataframe[column]  - dataframe[column + '_lag_' + str(i+1)])
        for i in range(horizon):
                  dataframe[column + '_horizon_' + str(i+1)] = dataframe[column].shift(-(i+1))

                  #dataframe['log_' + column + '_lag_' + str(i+1)] = np.log(dataframe[column + '_lag_' + str(i+1)],where=dataframe[column + '_lag_' + str(i+1)] > 0)
                  
                  dataframe['lag_difference_' + column + '_' + str(i+1)] = abs(dataframe[column]  - dataframe[column + '_lag_' + str(i+1)])


def prepara_dataset(dataframe, n_steps, horizon, k_features, casa,select=False):
   #Preparando os dados para a predição de série temporal
       
    #plt.figure()
    #plt.plot(dataframe.index, label='Tempo')
    #plt.xlabel('Tempo')
    #plt.ylabel('Valor')
    #plt.show()
    dataframe.columns = pd.io.parsers.base_parser.ParserBase({'names':dataframe.columns, 'usecols':None})._maybe_dedup_names(dataframe.columns)
    print(k_features)
    if k_features != 0:
      if 'RealPowerWatts_Circuit' in dataframe.columns:
          Y = dataframe['RealPowerWatts_Circuit']
          X = dataframe.drop('RealPowerWatts_Circuit', 1)
      else:
          Y = dataframe['Watts']
          X = dataframe.drop('Watts', 1)  

      x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)

      x_train = x_train.fillna(x_train.mean().fillna(0))
      y_train = y_train.fillna(y_train.mean()).to_frame()
      x_test = x_test.fillna(x_test.mean().fillna(0))
      y_test = y_test.fillna(y_test.mean()).to_frame()


      if select == True:
        selector = SelectKBest(mutual_info_regression, k=k_features)
        selector.fit(x_train, y_train)
        dump(selector, open('k_best_train_home' + casa + '.pkl', 'wb'))
        selector.fit(x_test, y_test)
        dump(selector, open('k_best_test_home' + casa + '.pkl', 'wb'))

      selector = load(open('k_best_train_home' + casa + '.pkl', 'rb'))
      cols = selector.get_support(indices=True)
      print(X.columns[cols].tolist())
      x_train = x_train.iloc[:,cols]
      selector = load(open('k_best_test_home' + casa + '.pkl', 'rb'))
      cols = selector.get_support(indices=True)
      x_test = x_test.iloc[:,cols]

      adiciona_lag(x_train,n_steps, horizon)
      adiciona_lag(x_test,n_steps, horizon)
      adiciona_lag(y_train,n_steps, horizon)
      adiciona_lag(y_test,n_steps, horizon)
      if 'RealPowerWatts_Circuit' in y_train.columns:
        columns = ['RealPowerWatts_Circuit']+dataframe.columns[dataframe.columns.str.contains('horizon')].to_list()
        x_train = pd.concat((x_train,y_train.drop(columns,1)),axis=1)
        y_train = y_train[columns]
        x_test = pd.concat((x_test,y_test.drop(columns,1)),axis=1)
        y_test = y_test[columns]
      else:
        columns = ['Watts']+dataframe.columns[dataframe.columns.str.contains('horizon')].to_list()
        x_train = pd.concat((x_train,y_train.drop(columns,1)),axis=1)
        y_train = y_train[columns]
        x_test = pd.concat((x_test,y_test.drop(columns,1)),axis=1)
        y_test = y_test[columns]
    else:
      if 'RealPowerWatts_Circuit' in dataframe.columns:
        aux =dataframe.index
        df = dataframe['RealPowerWatts_Circuit']
        df = df.to_frame()
        df.index = aux
      else:
        aux = dataframe.index
        df = dataframe['Watts']
        df = df.to_frame()
        df.index = aux
      
      adiciona_lag(df,n_steps,horizon)
      if 'RealPowerWatts_Circuit' in df.columns:
          columns = ['RealPowerWatts_Circuit']+df.columns[df.columns.str.contains('horizon')].to_list()
          Y = df[columns]
          X = df.drop(columns, 1)
      else:
          columns = ['Watts']+df.columns[df.columns.str.contains('horizon')].to_list()
          Y = df[columns]
          X = df.drop(columns, 1) 
      x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False, random_state=0)

     
    #print(dataframe.describe())

    x_train = x_train.fillna(x_train.mean().fillna(0))
    x_test = x_test.fillna(x_test.mean().fillna(0))
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scalers={}
    for column in x_train.columns:
      scaler = MinMaxScaler(feature_range=(0,1))
      s_s = scaler.fit_transform(x_train[column].values.reshape(-1,1))
      s_s=np.reshape(s_s,len(s_s))
      scalers['scaler_'+ column] = scaler
      x_train[column]=s_s

    for column in x_test.columns:
      scaler = MinMaxScaler(feature_range=(0,1))
      s_s = scaler.fit_transform(x_test[column].values.reshape(-1,1))
      s_s=np.reshape(s_s,len(s_s))
      scalers['scaler_'+column] = scaler
      x_test[column]=s_s
    
    y_train = target_scaler.fit_transform(y_train)
    y_test = target_scaler.fit_transform(y_test)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), target_scaler
                    
    

def carrega_home(root_path):
  root = root_path
  data = []
  all_directories = glob.glob(root+'*/')
  if '/home/nocs/TCC/Dataset/2013/homeC-all/homeC-generation/' in all_directories:
    all_directories.remove('/home/nocs/TCC/Dataset/2013/homeC-all/homeC-generation/')
  all_directories.sort()
  for dir in all_directories:

      all_files = [f for f in  glob.glob(dir + "/*.csv")]# if re.search(r'2012-Jun-[1](-p[1-2])?.csv', f)]
      print(dir)
      header = loadtxt(dir + "FORMAT",dtype=str,comments="#", delimiter=",", unpack=False)
      dfs = []
      dfs_p1 = []
      dfs_p2 = []
      for filename in all_files:

          start = filename.find('2')
          end = filename.find('.', start)
          df = pd.read_csv(filename, names=header)
          if 'TimestampUTC' in df.columns:
              df.set_index('TimestampUTC', inplace=True)
          else:
              df.set_index('TimestampLocal', inplace=True)
          df = df.loc[~df.index.duplicated(keep='first')]
          #df.index = df.index.map(lambda x : datetime.utcfromtimestamp(x).strftime('%m %d %H:%M %S'))
          #df.index = pd.to_datetime(df.index)


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
  if 'windDirectionDegrees' in df_home.columns:
      df_home['windDirectionDegrees'] = df_home['windDirectionDegrees'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
      df_home['windDirectionDegrees'] = df_home['windDirectionDegrees'].astype(float)
      df_home['windGustDirectionDegrees'] = df_home['windGustDirectionDegrees'].replace(r'\s+',np.nan,regex=True).replace('',np.nan)
      df_home['windGustDirectionDegrees'] = df_home['windGustDirectionDegrees'].astype(float)
  if 'Watts' in df_home.columns: 
    df_home['Watts'] = df_home['Watts'].replace('',np.nan).astype(float)
  df_home = df_home.select_dtypes(include=np.number)
  #df_home = df_home.resample('15T').sum()
  print("resample complete")

  return df_home

def load_data(path):
    
    df = carrega_home(path)
    #plt.figure()
    #plt.plot(df.index, label='Tempo')
    #plt.xlabel('Tempo')
    #plt.ylabel('Valor')
    #plt.show()
    #if 'RealPowerWatts_Circuit' in df.columns:
    #    aux = df.index
    #    df = df['RealPowerWatts_Circuit']
    #    df = df.to_frame()
    #    df.index = aux
    #else:
    #    aux = df.index
    #    df = df['Watts']
    #    df = df.to_frame()
    #    df.index = aux
    return df
    
  


def model_builder(hp):
  hp_filters1 = hp.Int('f1', min_value=32, max_value=128, step=32)
  hp_filters2 = hp.Int('f2', min_value=32, max_value=128, step=32)
  hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-128
  model = Sequential()
  
  model.add(Conv1D(64, kernel_size=1,activation='relu', input_shape=(10,1)))
  
  model.add(Conv1D(filters=hp_filters2, kernel_size=1, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  #model.add(Flatten())
  #model.add(RepeatVector(1))
  
  model.add(LSTM(hp_units, activation='tanh'))
  model.add(Dropout(0.2))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate,clipnorm=1),
            loss='mse',#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),#,#loss='mae',#tf.keras.losses.MeanSquaredError(),
            metrics='mae')

  return model
#x_train, x_test, y_train, y_test, target_scaler = load_data('/home/nocs/TCC/Dataset/2013/homeC-all/')
#print(y_train.shape[1])
#plt.figure()
#plt.plot(df['Watts'], label="Watts")
#plt.legend(loc="upper left")
#plt.xlabel('Time')
#plt.ylabel('Watts')
#plt.show()
#X_train.info()







