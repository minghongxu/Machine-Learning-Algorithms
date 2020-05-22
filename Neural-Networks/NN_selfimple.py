#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


input_list = 'downgesture_train.list'
test_file = 'downgesture_test.list'


# In[3]:


im = Image.open("gestures/C/C_down_5.pgm")


# In[4]:


arr = np.array(list(im.getdata()))


# In[20]:


class NeuralNetwork:
    def __init__(self, input_list, learning_rate = 0.1, epochs_numbers = 1000, inputlayer_size = 960, hiddenlayer_size = 100):
        self.learning_rate = learning_rate
        self.epochs_numbers = epochs_numbers
        self.inputlayer_size = inputlayer_size
        self.hiddenlayer_size = hiddenlayer_size
        self.input_list = input_list
        self.w1 = np.random.uniform(low=-0.01, high=0.01, size=(self.inputlayer_size,self.hiddenlayer_size))
        self.w2 = np.random.uniform(low=-0.01, high=0.01, size=(self.hiddenlayer_size,1))
    
    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def compute_xj(self,x0):
        s1 = np.dot(x0, self.w1)
        xj1 = self.sigmoid(s1)  #shape(1*100)
        s2 = np.dot(xj1, self.w2)
        xj2 = self.sigmoid(s2) #shape(1*1)
        return xj1,xj2
    
    def compute_delta(self,x0,y):
        xj1,xj2 = self.compute_xj(x0)
        delta1 = 2 * np.multiply((xj2 - y),self.sigmoid_derivative(xj2))  #shape(1*1)
        delta2 = np.multiply(np.dot(self.w2, delta1),self.sigmoid_derivative(xj1))  #shape(1*100)
        self.w1 -= self.learning_rate * np.dot(x0.reshape(len(x0),1), delta2.reshape(1,self.hiddenlayer_size))
        self.w2 -= self.learning_rate * np.dot(xj1.reshape(self.hiddenlayer_size,1), delta1.reshape(1,1))
    
    def propagate_everypoint(self):
        with open(self.input_list) as f:
            for row in f.readlines():
                row = row.strip('\n')
                im = Image.open(row)
                x = np.array(list(im.getdata()))/255
                if 'down' in row:
                    y = 1
                else:
                    y = 0
                self.compute_delta(x,y)
       
    def train(self):
        print('start trainning')
        for i in range(self.epochs_numbers):
            self.propagate_everypoint()
        print('finish trainning')
    
    def predict(self,test_file):
        result = []
        accurate = 0
        total = 0
        with open(test_file) as f:
            for row in f.readlines():
                total += 1
                row = row.strip('\n')
                im = Image.open(row)
                x = np.array(list(im.getdata()))/255
                tag = 0
                if 'down' in row:
                    tag = 1
                x1, y = self.compute_xj(x)
                if y >= 0.5:
                    result.append(1)
                    if tag == 1:
                        accurate += 1
                elif y < 0.5:
                    result.append(0)
                    if tag == 0:
                        accurate += 1
        return result, accurate/total
    


# In[21]:


n = NeuralNetwork(input_list, learning_rate = 0.1, epochs_numbers = 1000, inputlayer_size = 960, hiddenlayer_size = 100)


# In[22]:


#n.propagate_everypoint()


# In[23]:


n.train()


# In[24]:


predictions, accuracy = n.predict(test_file)


# 

# In[25]:


print(predictions)


# In[26]:


print(accuracy)


# In[ ]:




