
from sklearn.ensemble import RandomForestClassifier
from utils import generarEstadisticos


def randomForest(X,y):

    model = RandomForestClassifier()
    # Set the parameters by cross-validation
    tuned_parameters = [{'criterion': ['entropy'], 'n_estimators': [10,20,30,40,50,60,70,80,,90,100]},
                        {'criterion': ['gini'], 'n_estimators': [10,20,30,40,50,60,70,80,,90,100]}]

    return generarEstadisticos(model,X,y,tuned_parameters)
