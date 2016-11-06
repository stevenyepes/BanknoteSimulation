
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def generarEstadisticos(model, X, y, tuned_parameters):


    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Set the parameters by cross-validation
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, tuned_parameters, cv=10,
                           scoring='%s_macro' % score, n_jobs=-1)
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

    ## Retornar el modelo
    return clf

def reporte(model, X_validation, y_validation):
    print("#### Reporte del modelo con muestras para validación final ####")
    y_true, y_pred = y_validation, model.predict(X_validation)
    target_names = ['class 0', 'class 1']
    print(classification_report(y_true, y_pred, target_names=target_names))
    print()
