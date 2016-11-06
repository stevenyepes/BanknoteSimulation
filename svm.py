
from sklearn import svm
from utils import generarEstadisticos



def suportVectorMachine(X,y):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1,1e-2,1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'degree': [2, 3,4,5,6],
                                             'C': [1, 10, 100, 1000]}
                        ]
    model = svm.SVC()
    return generarEstadisticos(model,X,y,tuned_parameters)
