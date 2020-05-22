#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Gray + NN
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
data = pd.read_csv('train.csv')
label = data.label
data = data.drop('label',axis = 1)
data = data/ 255.0
train, test,train_labels, test_labels = train_test_split(data, label, train_size=0.8, random_state=42)
score=[]
fittime=[]
scoretime=[]
clf = MLPClassifier(solver='sgd', max_iter=1000, hidden_layer_sizes=(500,), learning_rate_init = 0.1, alpha = 0.1)
start_time = time.time()
clf.fit(train, train_labels.values.ravel())
fittime = time.time() - start_time
start_time = time.time()
score=clf.score(test,test_labels)
scoretime = time.time() - start_time
case1=['Gray Scale + NN',fittime,scoretime,score]
print(case1)

