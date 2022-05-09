import Smart_home
import sys
import flwr as fl
from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM 




class MicroGridClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        #print(f"Training finished for round {config['rnd']}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

def main() -> None:
    path = sys.argv[1]
    x_train, x_test, y_train, y_test = Smart_home.load_data(path)

    ts_inputs = tf.keras.Input(shape=(6,1))

    # units=10 -> The cell and hidden states will be of dimension 10.
    #             The number of parameters that need to be trained = 4*units*(units+2)
    x = LSTM(units=10)(ts_inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mse'])

    client = MicroGridClient(model,x_train, x_test, y_train, y_test)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()