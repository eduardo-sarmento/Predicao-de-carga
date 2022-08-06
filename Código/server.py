import flwr as fl
import Smart_home
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from matplotlib import pyplot as plt
from typing import Dict
import sys
import numpy as np

def get_eval_fn(model):
    diff = False
    n_steps = 10
    horizon =  1  
    k = 1

    df = Smart_home.load_data(path)

    x_train,x_test, y_train, y_test, target_scaler, features_scalers = Smart_home.prepara_dataset(df,n_steps, horizon, k,casa,diff=diff)

    def evaluate(parameters: fl.common.Weights):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        pred = model.predict(x_test)
        pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))
        y_test_rescaled =  target_scaler.inverse_transform(y_test)
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
        rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
        score = r2_score(y_test_rescaled, pred_rescaled)
        print('R-squared score para o conjunto de testes usando CNN-LSTM FEDERADO ' + str(k) +' features:', round(score,4))

        print('MAE CNN-LSTM FEDERADO ' + str(k) +' features:', mae)
        print('RMSE CNN-LSTM FEDERADO ' + str(k) +' features:', rmse)
        return loss, {"mae": accuracy}

    return evaluate


def fit_round(rnd: int) -> Dict:
    return {"rnd": rnd}

if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    path = sys.argv[1]
    aux = path.split('/')
    casa = aux[-2][4]
    server_eval = True


    if server_eval:
        df = Smart_home.load_data(path)
        diff = False
        n_steps = 10
        horizon =  1  
        k = 1

        x_train,x_test, y_train, y_test, target_scaler, features_scalers = Smart_home.prepara_dataset(df,n_steps, horizon, k,casa,diff=diff)

        model = Smart_home.CNN_LSTM_compile(x_train, y_train, horizon)
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            eval_fn=get_eval_fn(model),
            on_fit_config_fn=fit_round,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=3,
            on_fit_config_fn=fit_round,
        )
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 5})