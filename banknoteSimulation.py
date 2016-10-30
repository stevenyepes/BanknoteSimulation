import numpy as np
import svm as svm
import randomForest as rf
import kVecinos as kv
import redNeuronal as rn
import utils

# Obtener datos desde el archivo

BD = np.load('data.npy')
N = 1372

# Particiones

# Training 80%
a = BD[:int(N*0.8)]
# Validation 20%
b = BD[int(N*0.8):]

X = a[:,0:4]
y = np.array(a[:,4], dtype='int')

X_validation = b[:,0:4]
y_validation = np.array(b[:,4], dtype='int')

option = input("Ingrese por favor el modelo a entrenar, el sistema retornará\n" +
                "un análisis completo del resultado:\n" +
                "1: k-vecinos\n" +
                "2: Random Forest\n" +
                "3: Suport Vector Machines\n"+
                "4: Red Neuronal\n-> ")

if(option == "1"):
    model = kv.kVecinos(X,y)
elif(option == "2"):
    model = rf.randomForest(X,y)
elif(option == "3"):
    model = svm.suportVectorMachine(X,y)
elif(option == "4"):
    model = rn.neuralNetwork(X,y)
else:
    print("ingrese una opción válida")

utils.reporte(model, X_validation, y_validation)
