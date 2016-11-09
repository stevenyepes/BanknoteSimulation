from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
import numpy as np

from scipy.stats import pearsonr

def seleccionSecuencial(X,y):

    knn = KNeighborsClassifier(n_neighbors=4)
    # Sequential Forward Selection
    sfs = SFS(knn,
              k_features=4,
              forward=True,
              floating=False,
              scoring='accuracy',
              print_progress=False,
              cv=4,
              n_jobs=-1)
    sfs = sfs.fit(X, y)

    print('\nSequential Forward Selection (k=3):')
    print(sfs.k_feature_idx_)
    print('CV Score:')
    print(sfs.k_score_)
    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)


def fisher(X,y):
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    print(scores)

def pearson(X,y):
