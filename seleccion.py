from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


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

    print('\nSequential Forward Selection (k=4):')
    print(sfs.k_feature_idx_)
    print('CV Score:')
    print(sfs.k_score_)
    print(pd.DataFrame.from_dict(sfs.get_metric_dict()).T)
