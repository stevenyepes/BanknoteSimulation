from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.feature_selection import SelectPercentile,f_classif

def seleccionSecuencial(X,y):

    knn = KNeighborsClassifier(n_neighbors=4)
    # Sequential Forward Selection
    sfs = SFS(knn,
              k_features=4,
              forward=True,
              floating=False,
              scoring='accuracy',
              print_progress=True,
              cv=4,
              n_jobs=-1)
    sfs = sfs.fit(X, y)

    print('\nSequential Forward Selection (k=4):')
    print(sfs.k_feature_idx_)
    print('CV Score:')
    print(sfs.k_score_)
    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)


def fisher(X,y):
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X,y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    print(scores)

def corr(X1,y):
    clf = LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,solver='svd', store_covariance=True, tol=0.0001)
    clf.fit(X1, y)
    covar= clf.covariance_
    print(covar / covar.max(axis=0))
