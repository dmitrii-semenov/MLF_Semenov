#%%

# Load all libraries used
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, preprocessing
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Defined 3 points in 2D-space
X=np.array([[2, 1, 0],[4, 3, 0]])
# Calculate the covariance matrix:
R = np.matmul(X,X.T)/3
print(X)

# Calculate the SVD decomposition and new basis vectors
[U,D,V]=np.linalg.svd(R)
u1=U[:,0]
u2=U[:,1]
print(u1)
print(u2)

# Calculate the coordinates in new orthonormal basis
Xi1 = np.matmul(np.transpose(X),u1)
Xi2 = np.matmul(np.transpose(X),u2)

# Calculate the approximation of the original from new basis
Xaprox = np.matmul(u1[:,None],Xi1[None,:])+np.matmul(u2[:,None],Xi2[None,:])

# Load Iris dataset as in the last PC lab
iris=load_iris()
iris.feature_names
#print(iris.feature_names)
#print(iris.data[0:5,:])
#print(iris.target[:])

# Plot the first three colums in 3D
X=iris.data
y=iris.target
plt.figure()
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show

# Pre-processing (below are two scale options to choose from)
#Xscaler = StandardScaler()
Xscaler = MinMaxScaler()
Xpp=Xscaler.fit_transform(X)

# Plot the transformed feature space in 3D
plt.figure()
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpp[y==0,0],Xpp[y==0,1],Xpp[y==0,2],color='green')
axes2.scatter3D(Xpp[y==1,0],Xpp[y==1,1],Xpp[y==1,2],color='blue')
axes2.scatter3D(Xpp[y==2,0],Xpp[y==2,1],Xpp[y==2,2],color='magenta')
plt.show

# Define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())

# Plot the transformed feature space in 3D
plt.figure()
axes3=plt.axes(projection='3d')
axes3.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes3.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes3.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show
print(pca.explained_variance_) # we can reduce dimention that has
# a low value of variance (no useful information), otherwise mess

# Plot the transformed feature space in 2D
plt.figure()
plt.scatter(Xpca[y==0,0],Xpca[y==0,1],color='green')
plt.scatter(Xpca[y==1,0],Xpca[y==1,1],color='blue')
plt.scatter(Xpca[y==2,0],Xpca[y==2,1],color='magenta')
plt.show

# KNN clasifier on Xpp
X_train, X_test, y_train, y_test = train_test_split(Xpp,y,test_size=0.3)
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X_train,y_train)
Ypred1=knn1.predict(X_test)
# Import and show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test,Ypred1)

# KNN clasifier on Xpca
X_train, X_test, y_train, y_test = train_test_split(Xpca,y,test_size=0.3)
knn2=KNeighborsClassifier(n_neighbors = 3)
knn2.fit(X_train,y_train)
Ypred2=knn2.predict(X_test)
# Import and show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test,Ypred2)

# KNN clasifier on Xpca (feature 0,1)
X_train, X_test, y_train, y_test = train_test_split(Xpca[:,0:2],y,test_size=0.3)
knn3=KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train,y_train)
Ypred3=knn3.predict(X_test)
# Import and show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test,Ypred3)

# KNN clasifier on Xpca (feature 1,2)
X_train, X_test, y_train, y_test = train_test_split(Xpca[:,1:3],y,test_size=0.3)
knn3=KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train,y_train)
Ypred3=knn3.predict(X_test)
# Import and show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test,Ypred3)
#%%