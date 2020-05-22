#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Gray + PCA + NN
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
data = data/ 255.0
train, test,train_labels, test_labels = train_test_split(data, label, train_size=0.8, random_state=42)

# data[data>0]=data.applymap(lambda x: float(x/255))
# print(data['pixel100'])
sc = StandardScaler().fit(train)
s_train = sc.transform(train)
s_test = sc.transform(test)
#print(s_test.shape)
sk_pca = sklearnPCA().fit(s_train)
tr_pca = sk_pca.transform(s_train)
te_pca = sk_pca.transform(s_test)

perOfVariance = sk_pca.explained_variance_ratio_
perOfVc = sk_pca.explained_variance_ratio_.cumsum()
num_com = len(perOfVc[perOfVc<=0.9])
print('chose 90% attributes:', num_com)
sk_pca2 = sklearnPCA(n_components=num_com)
tr_pca2 = sk_pca2.fit_transform(s_train)
te_pca2 = sk_pca2.transform(s_test)
#print("Shape before PCA for Train: ",tr_pca.shape)
#print("Shape after PCA for Train: ",tr_pca2.shape)
# print("Shape before PCA for Test: ",te_pca.shape)
# print("Shape after PCA for Test: ",te_pca2.shape)

#model
clf = MLPClassifier(solver='sgd', max_iter=1000, hidden_layer_sizes=(500,), learning_rate_init = 0.1, alpha = 0.1)

score=[]
fittime=[]
scoretime=[]
start_time = time.time()
clf.fit(tr_pca2,train_labels.values.ravel())
fittime = time.time() - start_time
start_time = time.time()
score=clf.score(te_pca2,test_labels)
scoretime = time.time() - start_time
case3=['Gray + PCA + NN',fittime,scoretime,score]
print(case3)

