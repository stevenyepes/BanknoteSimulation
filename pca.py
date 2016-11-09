from sklearn.decomposition import PCA

def pca(X):
    print('Datos originales')
    print(X)
    pca = PCA(n_components=4)
    X_N= pca.fit_transform(X)
    print('Datos transformados')
    print(X_N)
    X_R = pca.inverse_transform(X_N)
    print('Datos restaurados')
    print(X_R)

    
