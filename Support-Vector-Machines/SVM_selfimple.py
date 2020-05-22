#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cvxopt
import sympy


# In[8]:


linfile = 'linsep.txt'
nonlinfile = 'nonlinsep.txt'


# In[9]:


lin_array = np.loadtxt(linfile, delimiter=',')
nonlin_array = np.loadtxt(nonlinfile, delimiter=',')


# In[ ]:





# In[10]:


class SVM:
    def __init__(self,input_array,whether_lin):
        self.array = input_array
        self.y = input_array[:,-1]
        self.x = input_array[:,:-1]
        self.size = input_array.shape[0]
        self.dim = input_array.shape[1] - 1
        self.whether_lin = whether_lin
        self.alpha = self.get_alpha()
        
    def kernal(self, x1, x2):
        return (1+1*np.dot(x1, x2.T))**2
        
    def get_alpha(self):
        temp1 = np.outer(self.y, self.y)
        if self.whether_lin:
            temp2 = np.dot(self.x, self.x.T)
        else:
            temp2 = self.kernal(self.x,self.x)
        Q = temp1 * temp2
        P = cvxopt.matrix(Q)
        q = cvxopt.matrix(np.ones(self.size)*-1)
        G = cvxopt.matrix(np.diag(np.ones(self.size) * -1))
        h = cvxopt.matrix(np.zeros(self.size))
        A = cvxopt.matrix(self.y.reshape((1, self.size)))
        b = cvxopt.matrix(0.0)
        result = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(result['x']).reshape(self.size,)
        for i in range(len(alpha)):
            if alpha[i] < 0.00001:
                alpha[i] = 0
        return alpha
    
    def get_parameter(self):
        if self.whether_lin:
            w = sum(self.alpha.reshape(self.size,1) * self.y.reshape(self.size,1) * self.x)
            index = np.where(self.alpha != 0)
            n = index[0][1]
            b = self.y[n] - np.dot(w.reshape(1,self.dim), self.x[n].reshape(self.dim, 1))
            return w, b
        else:
            index = np.where(self.alpha != 0)
            b = 0
            for i in index[0]:
                b += (self.y[i] - sum(self.alpha * self.y * self.kernal(self.x, self.x[i])))
            b = b/len(index[0])
            return b
    
    def curve_func(self,x):
        index = np.where(self.alpha != 0)
        b = 0
        for i in index[0]:
            b += (self.y[i] - sum(self.alpha * self.y * self.kernal(self.x, self.x[i])))
        b = b/len(index[0])
        
        y = sympy.Symbol('y')
        var = np.array([x,y]).reshape(1,2)
        f = self.kernal(self.x,var)
        f.reshape(100,)# +b
        s = sum(self.alpha * self.y * f.reshape(100,))
        result = sympy.solve([s + b], [y])
        l = []
        for i in result:
            num = complex(i[0])
            if num.imag == 0:
                l.append(i[0])
        return l
    
    def get_support_vectors(self):
        index = np.where(self.alpha != 0)
        return self.x[index]


# In[11]:


print('linear separable part')


# In[12]:


s = SVM(lin_array, True)


# In[13]:


w, b = s.get_parameter()


# In[14]:


print('w,b')
print(w,b)


# In[15]:


support_vectors = s.get_support_vectors()


# In[16]:


print('support_vectors:')
print(support_vectors)


# In[467]:


plt.scatter(lin_array[:,0], lin_array[:,1], c = lin_array[:,2])
plt.scatter(support_vectors[:,0], support_vectors[:,1])
x = np.linspace(0,1,10)
y = (-b-w[0]*x)/w[1]
plt.plot(x,y[0])
plt.show()


# In[ ]:





# In[5]:


print()


# In[17]:


print('nonlinear separable part')


# In[18]:


nl = SVM(nonlin_array, False)


# In[19]:


b_nl = nl.get_parameter()


# In[20]:


print('b')
print(b_nl)


# In[21]:


support_vectors_nl = nl.get_support_vectors()


# In[22]:


print('support_vectors:')
print(support_vectors_nl)


# In[460]:


plt.scatter(nonlin_array[:,0], nonlin_array[:,1], c = nonlin_array[:,2])
plt.scatter(support_vectors_nl[:,0], support_vectors_nl[:,1])

x = np.linspace(-20,20,200)
y1 = []
y2 = []
for i in x:
    y = nl.curve_func(i)
    if len(y) == 0:
        y1.append(None)
        y2.append(None)
    elif len(y)<2 and len(y)>0:
        y1.append(y[0])
        y2.append(y[0])
    else:
        y1.append(y[0])
        y2.append(y[1])
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()


# In[ ]:




