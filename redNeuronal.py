from sklearn import neural_network
from utils import generarEstadisticos

def neuralNetwork(X,y):
    model = neural_network.MLPClassifier()
    tuned_parameters = [{'solver':['lbfgs'], 'alpha':[1e-5],
                     'hidden_layer_sizes':[(5, 2),(7,5),(20,10),(30,15),(40,20)]},
                         ]
    return generarEstadisticos(model,X,y,tuned_parameters)
