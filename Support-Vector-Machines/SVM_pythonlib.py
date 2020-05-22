#!/usr/bin/env python
# coding: utf-8

# In[123]:


from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


# In[124]:


#readin Data
linfile = 'linsep.txt'
data = np.loadtxt(linfile, delimiter=',')
X=data[:,0:2]
Y=data[:,2]


# In[125]:


#Train Model
clf = svm.SVC(kernel='linear')
clf.fit(X,Y)


# In[126]:


#Get the equation
print('Intercept:')
print(clf.intercept_)
print('Weights:')
print(clf.coef_[0])


# In[127]:


print('support_vectors:')
print(clf.support_vectors_)


# In[132]:


#Plot
plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',alpha=1,s=50,edgecolors='k')
x2_lefttargeth = -(clf.coef_[0][0]*(-1)+clf.intercept_)/clf.coef_[0][1]
x2_righttargeth = -(clf.coef_[0][0]*(1)+clf.intercept_)/clf.coef_[0][1]
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],facecolors='none',s=100, edgecolors='yellow')
plt.plot([-1,1], [x2_lefttargeth,x2_righttargeth])
plt.show()


# In[133]:


#Readin Data
nonlinfile = 'nonlinsep.txt'
data = np.loadtxt(nonlinfile, delimiter=',')
X=data[:,0:2]
Y=data[:,2]


# In[138]:


#Train Model
clf = svm.SVC(kernel='poly',degree=2)
clf.fit(X, Y)  


# In[140]:


#Get the equation
print("Intercept:")
print(clf.intercept_)
print("Weights:")
print(clf.dual_coef_[0])


# In[141]:


print('support_vectors:')
print(clf.support_vectors_)


# In[ ]:




