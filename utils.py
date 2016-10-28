from sklearn.model_selection import cross_val_score
from sklearn import svm

def crossValidation(model,X, y):
    scores = cross_val_score(model, X, y, cv=5)
    #predicted = cross_val_predict(model, X_validation, y_validation, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)
