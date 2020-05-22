#!/usr/bin/env python
# coding: utf-8

# Group Members: Minghong Xu, Zhixin Xie
# ## Part 2:Software Familiarization

# ## PCA

# In[3]:


# Load libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# In[1]:


pca_file = r"pca-data.txt"


# In[10]:


df = np.loadtxt(pca_file, delimiter='\t')
df


# In[5]:


pca = PCA(n_components=2)


# In[8]:


pca.fit(df)
principalComponents = pca.transform(df)
principalDf = pd.DataFrame(data = principalComponents)


# In[9]:


print(principalDf)


# ## Fast Map

# We could not find any popular implementation or package of Fast Map.
