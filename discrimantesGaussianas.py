from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils import generarEstadisticos

def fdg(X,y):
    print("funciones discriminantes gaussianas")
    model = QuadraticDiscriminantAnalysis()
    tuned_parameters = [{'reg_param': [1,2,3,4,5,6,7,100]}]

    return generarEstadisticos(model,X,y,tuned_parameters)
