
from sklearn import neighbors
from utils import generarEstadisticos


def kVecinos(X,y):
    model = neighbors.KNeighborsClassifier()
    # Set the parameters by cross-validation
    tuned_parameters = [{'weights': ['uniform'], 'n_neighbors': [1,2,3,4,5,6,7,100]},
                        {'weights': ['distance'], 'n_neighbors': [1,2,3,4,5,6,7,100]}]
    return generarEstadisticos(model,X,y,tuned_parameters)
