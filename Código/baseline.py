import Smart_home
import sys
from matplotlib import pyplot as plt
from typing import Dict
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

if __name__ == "__main__":
    np.random.seed(0)
    path = sys.argv[1]
    aux = path.split('/')
    casa = aux[-2][4]
    #mixed_precision.set_global_policy('mixed_float16')
    df = Smart_home.load_data(path)

    n_steps = 10
    horizon =  0  
    k_features = [0,1,2,4,8]
    for k in k_features:
        
        x_train, x_test, y_train, y_test, target_scaler = Smart_home.prepara_dataset(df,n_steps, horizon, k)

#
        y_test_rescaled =  target_scaler.inverse_transform(y_test)
#
        best_iter = 100
        mlp = MLPRegressor(max_iter=best_iter,random_state=0)
        model_mlp = mlp.fit(x_train, y_train)
        pred = model_mlp.predict(x_test)
        pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))
        score = r2_score(y_test_rescaled, pred_rescaled)
        print('R-squared score para o conjunto de testes usando MLP ' + k +' features:', round(score,4))
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
##  #
        print('MAE MLP ' + k +' features:', mae)
##  #
        plt.figure()
        plt.plot(pred_rescaled, label="Predito")
        plt.plot(y_test_rescaled, label="Real")
        plt.legend(loc="upper left")
        plt.xlabel('Time')
        plt.ylabel('Watts')
        plt.title('Preditito vs. Real Consumo Eletrico MLP ' + k +' features para casa ' + casa)
        plt.savefig('Preditito vs. Real Consumo Eletrico MLP ' + k +' features para casa ' + casa +'.png')    

        tree = tree.DecisionTreeRegressor(random_state=0)

        model_tree = tree.fit(x_train, y_train)
        pred = model_tree.predict(x_test)
        pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))
        score = r2_score(y_test_rescaled, pred_rescaled)
        print('R-squared score para o conjunto de testes usando Decision Tree ' + k +' features:', round(score,4))
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
##  #
        print('MAE Decision Tree ' + k +' features:', mae)
##  #
        plt.figure()
        plt.plot(pred_rescaled, label="Predito")
        plt.plot(y_test_rescaled, label="Real")
        plt.legend(loc="upper left")
        plt.xlabel('Time')
        plt.ylabel('Watts')
        plt.title('Preditito vs. Real Consumo Eletrico Decision Tree com ' + k +' features para casa ' + casa)
        plt.savefig('Preditito vs. Real Consumo Eletrico Decision Tree com ' + k +' features para casa ' + casa +'.png')   

        forest = RandomForestRegressor(max_depth=2, random_state=0)
        model_forest = forest.fit(x_train, y_train)
        pred = model_forest.predict(x_test)
        pred_rescaled = target_scaler.inverse_transform(pred.reshape(-1, 1))
        score = r2_score(y_test_rescaled, pred_rescaled)
        print('R-squared score para o conjunto de testes usando Random Forest ' + k +' features:', round(score,4))
        mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
##  #
        print('MAE Random Forest ' + k +' features:', mae)
##  #
        plt.figure()
        plt.plot(pred_rescaled, label="Predito")
        plt.plot(y_test_rescaled, label="Real")
        plt.legend(loc="upper left")
        plt.xlabel('Time')
        plt.ylabel('Watts')
        plt.title('Preditito vs. Real Consumo Eletrico Random Forest com ' + k +' features para casa ' + casa)
        plt.savefig('Preditito vs. Real Consumo Eletrico Random Forest com ' + k +' features para casa ' + casa +'.png')   
##
    