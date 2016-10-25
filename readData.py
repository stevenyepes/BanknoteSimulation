import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
clf = tree.DecisionTreeClassifier()

# Obtener datos desde el archivo
BD = np.genfromtxt('data_banknote_authentication.txt', dtype='float', delimiter=',')

## guardar datos en archivo
## Desordenar los datos para luego dividirlos
N = 1372
np.random.shuffle(BD)

# Particiones

# Training 60%
a = BD[:int(N*0.6)]
# Validation 20%
b = BD[int(N*0.6):int(N*0.8)]
# Test 20%
c = BD[int(N*0.8):]

X = a[:,0:4]
y = np.array(a[:,4], dtype='int')

XTest = b[:,0:4]
yTest = np.array(b[:,4], dtype='int')

clf = clf.fit(X, y)

prediction = clf.predict(XTest)

print('Arbol de desición')
score = accuracy_score(yTest, prediction)
print(score,'\n')

print('Random Forest')
# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=25)
clf.fit(X, y)
prediction = clf.predict(XTest)
score = accuracy_score(yTest, prediction)
print(score,'\n')


print('K-neighbors')
n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)
prediction = clf.predict(XTest)
score = accuracy_score(yTest, prediction)
print(score,'\n')

print('SVC')
clf = svm.SVC(C=0.1)
clf.fit(X, y)
prediction = clf.predict(XTest)
score = accuracy_score(yTest, prediction)
print(score,'\n')
