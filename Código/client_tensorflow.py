import Smart_home
import sys
import flwr as fl
from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,ConvLSTM2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from matplotlib import pyplot as plt
import numpy as np



class SmartHomeClient(fl.client.NumPyClient):
    def __init__(self, model,x_train, x_test, y_train, y_test, casa, k_features):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.casa = casa
        self.k_features = k_features


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        es = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint('best_model'+ self.casa + '_' + str(self.k_features) + '_Federado.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit(self.x_train, self.y_train, epochs=100,validation_split=0.1,batch_size=32, verbose=0,callbacks=[es,checkpoint])
        self.model = load_model('best_model'+ self.casa + '_' + str(self.k_features) + '_Federado.h5')
        print(f"Training finished for round {config['rnd']}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"mae": accuracy}

def main() -> None:
    np.random.seed(0)
    tf.random.set_seed(0)
    path = sys.argv[1]
    aux = path.split('/')
    casa = aux[-2][4]
    df = Smart_home.load_data(path)
    diff = False
    n_steps = 10
    horizon =  1  
    k = 1

    x_train,x_test, y_train, y_test, target_scaler, features_scalers = Smart_home.prepara_dataset(df,n_steps, horizon, k,casa,diff=diff)
    
    model = Smart_home.CNN_LSTM_compile(x_train, y_train, horizon)

    client = SmartHomeClient(model,x_train, x_test, y_train, y_test,casa,k)
    fl.client.start_numpy_client("0.0.0.0:8080", client)
    pred = client.model.predict(x_test)

    pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))

    y_test_rescaled =  target_scaler.inverse_transform(y_test)
    if diff:
        pred_rescaled,y_test_rescaled = Smart_home.remove_lag(pred_rescaled,y_test_rescaled,x_test,features_scalers)
    mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
    rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
    score = r2_score(y_test_rescaled, pred_rescaled)
    print('R-squared score para o conjunto de testes usando CNN-LSTM FEDERADO ' + str(k) +' features:', round(score,4))

    print('MAE CNN-LSTM FEDERADO ' + str(k) +' features:', mae)
    print('RMSE CNN-LSTM FEDERADO ' + str(k) +' features:', rmse)

    plt.figure()
    plt.plot(pred_rescaled, label="Predito")
    plt.plot(y_test_rescaled, label="Real")
    plt.legend(loc="upper left")
    plt.xlabel('Tempo')
    plt.ylabel('Watts')
    plt.title('Preditito vs. Real Consumo Eletrico CNN-LSTM Federated Learning' + str(k) +' features para casa ' + casa)
    plt.savefig('Preditito vs. Real Consumo Eletrico CNN-LSTM Federated Learning ' + str(k) +' features para casa ' + casa +'.png')  


if __name__ == "__main__":
    main()