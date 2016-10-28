
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from utils import crossValidation


def randomForest(n):

    # Train uncalibrated random forest classifier on whole train and validation
    # data and evaluate on test data
    clf = RandomForestClassifier(n_estimators=n)
    return clf



def generarEstadisticos(model, XTest, yTest, X_validation, y_validation):
    print('Random Forest')
    crossValidation(model,XTest,yTest)
