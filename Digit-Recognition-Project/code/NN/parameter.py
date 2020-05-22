#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Binary + PCA + NN
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
data = data/ 255.0
train, test,train_labels, test_labels = train_test_split(data, label, train_size=0.8, random_state=42)

#Build the model
mlp = MLPClassifier(solver='sgd', max_iter=1000)

param_grid = {
    'hidden_layer_sizes': [(100,), (500,)],
    'learning_rate_init': [0.01,0.1],
    'alpha':[0.001, 0.01, 0.1]
}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(estimator=mlp,
                             param_grid=param_grid,
                             scoring='accuracy',
                             cv=3,
                             return_train_score=True, 
                             refit=True, 
                             n_jobs=-1)
clf.fit(train, train_labels.values.ravel())
print('Best Train Score：{}'.format(clf.best_score_))
print('Best Parameter:{}'.format(clf.best_params_))
print('Test Score：{}'.format(clf.score(test,test_labels)))

