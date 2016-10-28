import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import svm as svm
import randomForest as rf
import kVecinos as kv
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


model = svm.suportVectorMachine(1)
svm.prueba(model,X,y)
svm.generarEstadisticos(model, X, y, X_validation, y_validation)

model = kv.kVecinos(15)
kv.generarEstadisticos(model, X, y, X_validation, y_validation)
kv.prueba(model,X,y)

model = rf.randomForest(25)
rf.generarEstadisticos(model, X, y, X_validation, y_validation)
