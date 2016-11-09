import numpy as np
from sklearn.model_selection import train_test_split
import svm as svm
import randomForest as rf
import kVecinos as kv
import redNeuronal as rn
import discrimantesGaussianas as dg
import utils
from seleccion import seleccionSecuencial
from pca import pca

# Obtener datos desde el archivo
BD = np.load('data.npy')
# Separar los datos en variables y salida
X = BD[:,0:4]
y = np.array(BD[:,4], dtype='int')
# Particiones
# Training 80% Validation 20%
X, X_validation, y, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=0)

## Menú
option = input("###      Banknote Authentication     ###\n" +
                " Ingrese por favor el modelo a entrenar, el sistema retornará\n" +
                " un análisis completo del resultado: \n" +
                " 1: k-vecinos\n" +
                " 2: Random Forest\n" +
                " 3: Suport Vector Machines\n"+
                " 4: Red Neuronal\n"+
                " 5: Funciones discriminantes gaussianas\n"+
                " 6: Selección secuencial\n-> ")

if(option == "1"):
    model = kv.kVecinos(X,y)
elif(option == "2"):
    model = rf.randomForest(X,y)
elif(option == "3"):
    model = svm.suportVectorMachine(X,y)
elif(option == "4"):
    model = rn.neuralNetwork(X,y)
elif(option == "5"):
    model = dg.fdg(X,y)
elif(option == "6"):
    seleccionSecuencial(X,y)
elif(option == "7"):
    pca(X)    
else:
    print("ingrese una opción válida")

## Reporte final
try:
    utils.reporte(model, X_validation, y_validation)
except NameError as e:
    pass
