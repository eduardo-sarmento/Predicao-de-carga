from cProfile import label
import pandas as pd
from matplotlib import pyplot as plt
import sys

def main():
    df_A = pd.read_csv("/home/nocs/TCC/Código/monitor A.txt",sep=' ',index_col="Time")
    df_A.index = pd.to_datetime(df_A.index, format="%H:%M:%S")
    df_B = pd.read_csv("/home/nocs/TCC/Código/monitor B.txt",sep=' ',index_col="Time")
    df_B.index = pd.to_datetime(df_B.index, format="%H:%M:%S")
    df_C = pd.read_csv("/home/nocs/TCC/Código/monitor C.txt",sep=' ',index_col="Time")
    df_C.index = pd.to_datetime(df_C.index, format="%H:%M:%S")
    fases_A = ["14:45:24", 
                "14:48:54",
                "14:52:13", 
                "14:56:08", 
                "15:02:24", 
                "15:05:55"] 

    fases_B = ["14:52:13", 
                "14:57:20", 
                "14:57:22", 
                "14:59:38", 
                "14:59:44", 
                "15:01:53"] 

    fases_C = ["14:45:25", 
                "14:52:11", 
                "14:57:22",
                "14:59:42", 
                "14:59:44", 
                "15:02:22", 
                "15:02:24", 
                "15:04:30"] 

    for column in df_A:

        ax = df_A.reset_index().plot(x='Time',y=column,title=column + ' de cada Cliente')
        fig = ax.get_figure()
        df_B.reset_index().plot(ax=ax,x='Time',y=column)
        df_C.reset_index().plot(ax=ax,x='Time',y=column)
        #for fase in fases_A:
        #    ax.axvline(x=pd.to_datetime(fase, format="%H:%M:%S"), color='k', linestyle='--')
        #for fase in fases_B:
        #    ax.axvline(x=pd.to_datetime(fase, format="%H:%M:%S"), color='k', linestyle='--')
        #for fase in fases_C:
        #    ax.axvline(x=pd.to_datetime(fase, format="%H:%M:%S"), color='k', linestyle='--')
        ax.legend(["Cliente A", "Cliente B","Cliente C"])
        ax.set(xlabel="Tempo", ylabel=column)
        fig.savefig(column + ' de cada Cliente.png')


if __name__ == "__main__":
    main()
    