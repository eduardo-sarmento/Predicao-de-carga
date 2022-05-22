import Smart_home
import sys
import flwr as fl
from collections import OrderedDict
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


class SmartHomeClient(fl.client.NumPyClient):
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
        self.model.fit(self.x_train, self.y_train, epochs=300, batch_size=128, steps_per_epoch=3, verbose=0)
        print(f"Training finished for round {config['rnd']}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

def main() -> None:
    path = sys.argv[1]
    aux = path.split('/')
    x_train, x_test, y_train, y_test, target_scaler = Smart_home.load_data(path)
    
    x_train = x_train[...,None]
    x_test = x_test[...,None]

    checkpoint = ModelCheckpoint("/home/nocs/TCC/Código", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    EarlyStopping(monitor='val_loss', patience=10)

    # units=10 -> The cell and hidden states will be of dimension 10.
    #             The number of parameters that need to be trained = 4*units*(units+2)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(y_train.shape[1]))
    model.add(LSTM(units=192, activation='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01,clipnorm=1),
              loss='mse',#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),#,#loss='mae',#tf.keras.losses.MeanSquaredError(),
              metrics='mae')

    client = SmartHomeClient(model,x_train, x_test, y_train, y_test)
    fl.client.start_numpy_client("0.0.0.0:8080", client)
    pred = client.model.predict(x_test[:,:,0])
    pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))
    y_test_rescaled =  target_scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
    score = r2_score(y_test_rescaled, pred_rescaled)
    print('R-squared score para o conjunto de testes usando CNN-LSTM:', round(score,4))

    print("MAE CNN-LSTM: ", mae)

       ## Plot all predictions
    plt.figure()
    plt.plot(pred_rescaled, label="Predito")
    plt.plot(y_test_rescaled, label="Real")
    plt.legend(loc="upper left")
    plt.xlabel('Tempo')
    plt.ylabel('Watts')
    plt.title('Preditito vs. Real Consumo Eletrico CNN-LSTM Federated Learning')
    #plt.show()
    plt.savefig('Preditito vs. Real Consumo Eletrico CNN-LSTM Federated Learning ' + aux[-2] +'.png')  


if __name__ == "__main__":
    main()