from sklearn import neural_network
from utils import generarEstadisticos

def neuralNetwork(X,y):
    model = neural_network.MLPClassifier()
    tuned_parameters = [{'hidden_layer_sizes': [(100, 50)]},
                        {'hidden_layer_sizes': [(100,40)]} ]
    return generarEstadisticos(model,X,y,tuned_parameters)
