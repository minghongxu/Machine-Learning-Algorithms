#!/usr/bin/env python
# coding: utf-8

# ## Part 2:Software Familiarization

# Group Members: Minghong Xu, Zhixin Xie

# ## K-means

# In[211]:


# Load libraries
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# In[212]:


txt_file = r"clusters.txt"


# In[213]:


df = pd.read_table(txt_file,sep = ',' , header = None)
df.columns = ['V1','V2']
df.head()


# In[214]:


f1 = df['V1'].values
f2 = df['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)


# In[215]:


kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.fit(X).predict(X)
centroids = kmeans.cluster_centers_


# In[216]:


print(centroids) 


# In[217]:


plt.scatter(f1,f2, c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='red')


# ## Gaussian Mixture Model (GMM)

# In[218]:


from sklearn import mixture


# In[219]:


gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X)
labels = gmm.predict(X)
plt.scatter(f1, f2, c=labels, s=40, cmap='viridis');


# In[220]:


print(gmm.weights_)


# In[221]:


print(gmm.means_)


# In[222]:


print(gmm.covariances_)

