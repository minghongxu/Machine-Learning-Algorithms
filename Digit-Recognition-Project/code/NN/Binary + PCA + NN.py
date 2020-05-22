#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Binary + PCA + NN
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import sklearn
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
data[data>0]=1

train, test,train_labels, test_labels = train_test_split(data, label, train_size=0.8, random_state=42)
standard_data = StandardScaler().fit(train)
tr_data = standard_data.transform(train)
te_data = standard_data.transform(test)
sk_pca = sklearnPCA().fit(tr_data)
perOfvar = sk_pca.explained_variance_ratio_
perOfvarc = sk_pca.explained_variance_ratio_.cumsum()

num_com = len(perOfvarc[perOfvarc<=0.9])
print('keep ', num_com,' components')
sk_pca2 = sklearnPCA(n_components=217)
tr_pca2 = sk_pca2.fit_transform(tr_data)
te_pca2 = sk_pca2.transform(te_data)

clf = MLPClassifier(solver='sgd', max_iter=1000, hidden_layer_sizes=(500,), learning_rate_init = 0.1, alpha = 0.1)
#Binary Images and PCA Reduction
start_time = time.time()
clf.fit(tr_pca2,train_labels.values.ravel())
fittime = time.time() - start_time
start_time = time.time()
score=clf.score(te_pca2,test_labels)
scoretime = time.time() - start_time
case4=['Binary + PCA + NN',fittime,scoretime,score]
print(case4)

