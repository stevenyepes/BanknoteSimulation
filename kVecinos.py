
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from utils import crossValidation


def kVecinos(n_neighbors):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    return clf


def generarEstadisticos(model,X,y):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'weights': ['uniform'], 'n_neighbors': [1,2,3,4,5,6,7,100]},
                        {'weights': ['distance'], 'n_neighbors': [1,2,3,4,5,6,7,100]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, tuned_parameters, cv=10,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        target_names = ['class 0', 'class 1']
        print(classification_report(y_true, y_pred, target_names=target_names))
        print()
