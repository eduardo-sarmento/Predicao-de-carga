import flwr as fl
import Smart_home
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from typing import Dict
import sys
import numpy as np


# units=10 -> The cell and hidden states will be of dimension 10.
#             The number of parameters that need to be trained = 4*units*(units+2)


def fit_round(rnd: int) -> Dict:
#    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model):
#    """Return an evaluation function for server-side evaluation."""

    df = Smart_home.load_data('/home/nocs/TCC/Dataset/2013/homeB-all/')
    n_steps = 10
    horizon =  0  
    k = 0
    _,X_test,_,y_test , _, _ = Smart_home.prepara_dataset(df,n_steps, horizon, k,'B')

    def evaluate(parameters: fl.common.Weights):
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

if __name__ == "__main__":
    #path = sys.argv[1]
    #aux = path.split('/')
    #casa = aux[-2][4]
    np.random.seed(0)
    tf.random.set_seed(0)
    #df = Smart_home.load_data('/home/nocs/TCC/Dataset/2013/homeB-all/')
    n_steps = 10
    horizon =  0  
    k = 0
    #x_train,_ , y_train,_ , target_scaler, features_scalers = Smart_home.prepara_dataset(df,n_steps, horizon, k,'B')
    #model = Smart_home.CNN_LSTM_compile(x_train, y_train, horizon)

    
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
       # eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 5})