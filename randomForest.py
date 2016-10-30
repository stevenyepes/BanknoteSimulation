
from sklearn.ensemble import RandomForestClassifier
from utils import generarEstadisticos


def randomForest(X,y):

    model = RandomForestClassifier()
    # Set the parameters by cross-validation
    tuned_parameters = [{'criterion': ['entropy'], 'n_estimators': [1,2,3,4,5,6,7,100]},
                        {'criterion': ['gini'], 'n_estimators': [1,2,3,4,5,6,7,100]}]

    return generarEstadisticos(model,X,y,tuned_parameters)
