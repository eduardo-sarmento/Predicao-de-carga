import pandas as pd
from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":

    #path = sys.argv[1]
    #aux = path.split(' ')
    #print(aux)
    casa = "B"
    df = pd.read_csv("/home/nocs/TCC/Código/Result B")
    print(df.describe())
    vertical = [8,14,20,26,30,33,35]
    
    plt.figure()
    plt.plot(df['MAE'])
    for i in vertical:
        plt.axvline(x=i,color ='red',linestyle="dashed")
    plt.title('MAE para casa ' + casa)
    plt.ylabel('MAE')
    plt.xlabel('combinacao')
    plt.savefig('MAE para casa ' + casa) 
    plt.figure()

    plt.figure()
    plt.plot(df['R2'])
    for i in vertical:
        plt.axvline(x=i,color ='red',linestyle="dashed")
    plt.title('R2 score para casa ' + casa)
    plt.ylabel('R2 score')
    plt.xlabel('combinacao')
    plt.savefig('R2 score para casa ' + casa) 
    plt.figure()


    plt.figure()
    plt.plot(df['RMSE'])
    for i in vertical:
        plt.axvline(x=i,color ='red',linestyle="dashed")
    plt.title('RMSE para casa ' + casa)
    plt.ylabel('RMSE')
    plt.xlabel('combinacao')
    plt.savefig('RMSE para casa ' + casa) 
    plt.figure()
