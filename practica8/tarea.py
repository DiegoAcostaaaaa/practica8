import urllib.request
import numpy as np

#urllib.request.urlretrieve('https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv', 'diabetes.txt')

diabetes_data = np.genfromtxt('diabetes.txt', delimiter=',', skip_header=1) 
print("El contenido del dataset: \n", diabetes_data)
print("Las dimensiones del arreglo: \n", diabetes_data.shape)

pesos=np.array([1,1,1,1,1,1,1,1,1])
print("El contenido del arreglo de pesos: \n", pesos)
print("Las dimensiones del arreglo: \n", pesos.shape)

diabetes_modificado = diabetes_data @ pesos
print("El arreglo con los resultados de la multiplicación: \n", diabetes_modificado)
print("Las dimensiones del arreglo con los resultados\n ", diabetes_modificado.shape)

diabetes_results = np.concatenate((diabetes_data, diabetes_modificado.reshape(768, 1)), axis=1)
print("Resultado de concatenarle una columna más al arreglo: \n", diabetes_results)
np.savetxt('diabetes_results.txt',diabetes_results,fmt='%.2f',delimiter=',',header='Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome,Resultado',comments='')

print('Suma todos los elementos de la columna de edad:',diabetes_data[:,7].sum())

print('Promedio de Edad del dataset:',diabetes_data[:,7].mean())
print('Desviacion estandar de la Edad del dataset:',diabetes_data[:,7].std())
print('Mayor Edad del dataset:',diabetes_data[:,7].max())
print('Menor Edad del dataset:',diabetes_data[:,7].min())

print('Dataset mas 1:\n',diabetes_data+1)
print('Dataset por 2:\n',diabetes_data*2)
print('Dataset entre 10:\n',diabetes_data/10)

#Broadcasting
arreglo=np.array([10,11,12,13,14,15,16,17,18])
print('Le sumamos el dataset un arreglo, con broadcasting:\n',diabetes_data+arreglo)

#Booleanos
print('Arreglo del la coincidencia del dataset original y el dataset por 2:\n',diabetes_data==diabetes_data*2)

#Partir
print('La edad de la primera fila',diabetes_data[0,7])
print('Todas las edades:\n',diabetes_data[:,7])