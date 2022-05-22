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



# units=10 -> The cell and hidden states will be of dimension 10.
#             The number of parameters that need to be trained = 4*units*(units+2)


def fit_round(rnd: int) -> Dict:
#    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model):
#    """Return an evaluation function for server-side evaluation."""

    _,X_test, _, y_test = Smart_home.load_data('/home/nocs/TCC/Dataset/2013/homeB-all/')

    def evaluate(parameters: fl.common.Weights):
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

if __name__ == "__main__":
    #x_train,_, y_train, _ = Smart_home.load_data('/home/nocs/TCC/Dataset/2013/homeB-all/')
    #model = Sequential()
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    #model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(RepeatVector(y_train.shape[1]))
    #model.add(LSTM(units=512, activation='tanh',return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(TimeDistributed(Dense(100, activation='relu')))
    #model.add(TimeDistributed(Dropout(0.2)))
    #model.add(TimeDistributed(Dense(1)))
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,clipnorm=1),
    #          loss='mse',#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),#,#loss='mae',#tf.keras.losses.MeanSquaredError(),
    #          metrics='mae')
    
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
    #    eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 3})