from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from utils import generarEstadisticos

def fdg(X,y):
    print("funciones discriminantes gaussianas")
    model = QuadraticDiscriminantAnalysis()
    tuned_parameters = [{'reg_param': [0.0,0.001,0.01,0.1,1]}]

    return generarEstadisticos(model,X,y,tuned_parameters)
