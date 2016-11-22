from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import kVecinos as kv

def pca(X, y,componentes):
    print('Datos originales')
    print(X)
    pca = PCA(n_components=componentes)
    X_N= pca.fit_transform(X)
    print('Datos transformados')
    print(X_N)
    print('Matriz de varianza explicada')
    plt.plot(pca.explained_variance_ratio_)
    title = 'Porcentaje de varianza explicada con ' + str(componentes) + ' componentes'
    plt.title(title)
    plt.show()
    model = kv.kVecinos(X_N, y)
