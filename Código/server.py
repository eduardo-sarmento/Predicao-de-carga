import flwr as fl
import Smart_home
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM 
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
    
    ts_inputs = tf.keras.Input(shape=(6,1))
    x = LSTM(units=10)(ts_inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse'])

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3})