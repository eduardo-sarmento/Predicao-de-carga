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
from tensorflow.keras.layers import Dense,Dropout,LSTM 
import tensorflow_addons as tfa
from tensorflow_addons.rnn import LayerNormLSTMCell
from statsmodels.tsa import stattools
from matplotlib import pyplot as plt



def prepara_dataset(dataframe, n):
   #Preparando os dados para a predição de série temporal
   #Assumi que o valor atual de potencia pode ser previsto usando os n-1 ultimos valores da potencia
    dataframe.columns = pd.io.parsers.base_parser.ParserBase({'names':dataframe.columns, 'usecols':None})._maybe_dedup_names(dataframe.columns)
    for column in dataframe.columns:
    #Aplicando logaritmo na potencia real
        #if dataframe[column].dtypes == np.float64:
        #      dataframe['log_' + column] = np.log(dataframe[column],where=dataframe[column] > 0) 

        #Loop que calcula os valores da diferença entre a portencia atual, a potencia anterior e a potencia anterior da anterior
              for i in range(n):
                
                  dataframe[column + '_lag_' + str(i+1)] = dataframe[column].shift(i+1)

                  #dataframe['log_' + column + '_lag_' + str(i+1)] = np.log(dataframe[column + '_lag_' + str(i+1)],where=dataframe[column + '_lag_' + str(i+1)] > 0)
                  
                  #dataframe['log_difference_' + column + '_' + str(i+1)] = dataframe['log_' + column]  - dataframe[column + '_lag_' + str(i+1)]
    

def carrega_home(root_path):
  root = root_path
  data = []
  all_directories = glob.glob(root+'*/')
  all_directories.sort()
  for dir in all_directories:

      all_files = [f for f in  glob.glob(dir + "/*.csv")] #if re.search(r'2012-May-[1-7](-p[1-2])?.csv', f)]
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
          #for column in df:
          #if df[column].dtype is "float64":
          #df[column] = df[column].astype('float32')
          #else:
          #df[column] = df[column].astype('str')
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
  return df_home

def load_data(path):
    df = carrega_home(path)
    if 'RealPowerWatts_Circuit' in df.columns:
        aux = df.index
        df = df['RealPowerWatts_Circuit']
        df = df.to_frame()
        df.index = aux
    else:
        aux = df.index
        df = df['Watts']
        df = df.to_frame()
        df.index = aux
    df.sort_index(inplace=True)
    df.fillna(df.mean().round(1), inplace=True)
    #acf_djia, confint_djia, qstat_djia, pvalues_djia = stattools.acf(df,
    #                                                         nlags=50,
    #                                                         qstat=True,
    #                                                         fft=True,
    #                                                         alpha = 0.05)
    #aux = pd.Series(acf_djia)
    #plt.figure(figsize=(7, 5))
    #plt.plot(pd.Series(acf_djia), color='r', linewidth=2)
    #plt.title('Autocorrelation of Bitcoin Closing Price', weight='bold', fontsize=16)
    #plt.xlabel('Lag', weight='bold', fontsize=14)
    #plt.ylabel('Value', weight='bold', fontsize=14)
    #plt.xticks(weight='bold', fontsize=12, rotation=45)
    #plt.yticks(weight='bold', fontsize=12)
    #plt.grid(color = 'y', linewidth = 0.5)
    #plt.show()
    prepara_dataset(df,10)
    df.applymap(lambda x: np.log(x,where=x>0))
    #df.info()
    
    return df

def model_builder(hp):
  model = keras.Sequential()
  #model.add(keras.layers.Flatten(input_shape=(6, 1)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-128
  hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
  model.add(LSTM(units=hp_units, return_sequences=True,input_shape=(5,1)))#(ts_inputs))
  #lnLSTMCell = tfa.rnn.LayerNormLSTMCell(units=hp_units,input_shape=(5,1))
  #rnn = tf.keras.layers.RNN(lnLSTMCell, return_sequences=True, return_state=True)
  #model.add(rnn)#(ts_inputs))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='linear'))
  #x = (x)
  #outputs = Dense(1, activation='linear')(x)
  #model = tf.keras.Model(inputs=(6,1),outputs=outputs)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate,clipnorm=1),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse'])

  return model
#df = load_data('/home/nocs/TCC/Dataset/2013/homeB-all/')
#X_train.info()







