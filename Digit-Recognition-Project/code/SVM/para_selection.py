import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# binary image + SVM

data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
data[data>0]=1

train, test,train_labels, test_labels = train_test_split(data[:20000], label[:20000], train_size=0.6, random_state=42)
standard_data = StandardScaler().fit(train)
tr_data = standard_data.transform(train)
te_data = standard_data.transform(test)
sk_pca = sklearnPCA().fit(tr_data)
perOfvar = sk_pca.explained_variance_ratio_
perOfvarc = sk_pca.explained_variance_ratio_.cumsum()


sk_pca2 = sklearnPCA(n_components=217)
tr_pca2 = sk_pca2.fit_transform(tr_data)
te_pca2 = sk_pca2.transform(te_data)



grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10,100], "gamma":[1, 0.1, 0.01,0.001]}, cv=5)
grid.fit(tr_pca2,train_labels.values.ravel())
print("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))

