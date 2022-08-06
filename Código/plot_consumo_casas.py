from matplotlib import pyplot as plt
import sys
import Smart_home

casaA = Smart_home.load_data("/home/nocs/TCC/Dataset/2013/homeA-all/")
casaB = Smart_home.load_data("/home/nocs/TCC/Dataset/2013/homeB-all/")
casaC = Smart_home.load_data("/home/nocs/TCC/Dataset/2013/homeC-all/")

fig = plt.figure()
plt.plot(casaA['Watts'])
plt.title('Média do consumo de energia de 15 em 15 minutos para casa A')
plt.ylabel('Watts')
plt.xlabel('Tempo')
fig.autofmt_xdate()
plt.savefig('Consumo casa A') 

fig = plt.figure()
plt.plot(casaB['Watts'])
plt.title('Média do consumo de energia de 15 em 15 minutos para casa B')
plt.ylabel('Watts')
plt.xlabel('Tempo')
fig.autofmt_xdate()
plt.savefig('Consumo casa B') 

fig = plt.figure()
plt.plot(casaC['Watts'])
plt.title('Média do consumo de energia de 15 em 15 minutos para casa C')
plt.ylabel('Watts')
plt.xlabel('Tempo')
fig.autofmt_xdate()
plt.savefig('Consumo casa C') 