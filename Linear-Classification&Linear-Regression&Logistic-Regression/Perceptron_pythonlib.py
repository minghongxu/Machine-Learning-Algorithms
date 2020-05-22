#!/usr/bin/env python
# coding: utf-8

# ## Part 2:Software Familiarization

# ## Linear Classification

# In[182]:


#loading libaray
import numpy as np
from sklearn import linear_model


# In[183]:


classification_file = 'classification.txt'


# In[184]:


classification_array = np.loadtxt(classification_file, delimiter=',')


# ### PLA

# In[185]:


#loading libaray
from sklearn.linear_model import Perceptron


# In[186]:


X,y = classification_array[:,:3], classification_array[:, 3]
X = np.c_[np.ones(len(X)), np.array(X)]


# In[187]:


#pla
pla = Perceptron()
model_pla = pla.fit(X, y)
pla_parameters = model_pla.coef_
score = pla.score(X,y)


# In[188]:


print(score)
print(pla_parameters)


# ### Pocket

# In[205]:


X,y = classification_array[:,:3], classification_array[:, 4]
X = np.c_[np.ones(len(X)), np.array(X)]


# In[206]:


pla = Perceptron()
pla.warm_start = True
best_score = 0
for i in range (0, 7000):  
    pla = pla.fit(X, y)
    score = pla.score(X, y)
    if (best_score <= score or i == 0):
        best_score = score
        param = pla.coef_ 


# In[207]:


print(best_score)
print(param)


# ## Linear Regression

# In[192]:


#loading libaray
import numpy as np
from sklearn import linear_model


# In[193]:


linear_file = 'linear-regression.txt'
linear_array = np.loadtxt(linear_file, delimiter=',')


# In[194]:


XY,z = linear_array[:,:2], linear_array[:,2]


# In[195]:


model_lin = linear_model.LinearRegression()
model_lin.fit(XY,z)
predicted_classes = model_lin.predict(XY)
parameters = model_lin.coef_


# In[196]:


print(parameters)


# ## Logistic Regression

# In[197]:


# Load libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[198]:


X,y = classification_array[:, :3], classification_array[:, 4]
X = np.c_[np.ones(len(X)), np.array(X)]
print(X)


# In[199]:


model_log = LogisticRegression()
model_log.fit(X, y)
score = model_log.score(X,y)
parameters = model_log.coef_


# In[200]:


print(score)
print(parameters)

