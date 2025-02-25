#%%

# Load all libraries used
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.datasets import load_iris, make_blobs
import matplotlib.pyplot as plt
import numpy as np
from numpy import quantile, where, random

# Load IRIS datasets
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[:,:])
print(iris.target)

# Divide into testing and training parts
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Support Vector Machine initialization
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()
SVMmodel.score(X_test,y_test)
#print(SVMmodel.get_params())
print(SVMmodel.score(X_test,y_test))

# First two features
X = iris.data[:,0:2]
X.shape
#print(X.shape)

# Visualization
plt.figure()
plt.scatter(X[y==0,0], X[y==0,1],color='blue')
plt.scatter(X[y==1,0], X[y==1,1],color='red')
#plt.scatter(X[y==2,0], X[y==2,1],color='cyan')
plt.show()

# Train only on two classes
X1=iris.data[iris.target!=2,0:2]
y1=iris.target[iris.target!=2]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.1)
print(X1_train.shape)
print(X1_test.shape)
print(y1_train.shape)
print(y1_test.shape)

# Support Vector Machine for only two classes
SVMmodel1=SVC(kernel='linear',C=200)
SVMmodel1.fit(X1_train,y1_train)
SVMmodel1.get_params()
SVMmodel1.score(X1_test,y1_test)
#print(SVMmodel.get_params())
print(SVMmodel1.score(X1_test,y1_test))


# Visualization for two classes
plt.figure()
plt.scatter(X1[y1==0,0], X1[y1==0,1],color='blue')
plt.scatter(X1[y1==1,0], X1[y1==1,1],color='red')
#plt.scatter(X[y==2,0], X[y==2,1],color='cyan')

# Draw a separating line
upvectors=SVMmodel1.support_vectors_
W=SVMmodel1.coef_
b=SVMmodel1.intercept_
print(W)
print(b)
linex = np.linspace(np.min(X1[:,0]),np.max(X1[:,0]),100)
liney = -b/W[0,1] - W[0,0]/W[0,1]*linex
plt.scatter(linex,liney,color = 'black')
plt.show()

# One-class clasifier
random.seed(11)
x, _ = make_blobs(n_samples=300, centers=1, cluster_std=.3, center_box=(4, 4))
plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.show()

# Train one-class SVM
SVMmodelOne = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
SVMmodelOne.fit(x)
pred = SVMmodelOne.predict(x)
anom_index = where(pred==-1)
values = x[anom_index]

plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()

# Custom outliers marking (threshold control)
scores = SVMmodelOne.score_samples(x)
thresh = quantile(scores, 0.01)
print(thresh)
index = where(scores<=thresh)
values = x[index]

plt.figure()
plt.scatter(x[:,0], x[:,1])
plt.scatter(values[:,0], values[:,1], color='red')
plt.axis('equal')
plt.show()
#%%
