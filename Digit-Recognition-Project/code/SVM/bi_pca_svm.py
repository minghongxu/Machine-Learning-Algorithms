import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import sklearn
from sklearn.model_selection import GridSearchCV

# binary image + SVM

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


plt.figure(figsize=(20,10))
ind = np.arange(len(perOfvar))
plt.bar(ind,perOfvar,color = 'green')
plt.xlabel('n_components')
plt.ylabel('Variance')
plt.show()

num_com = len(perOfvarc[perOfvarc<=0.9])
print('keep ', num_com,' components')
sk_pca2 = sklearnPCA(n_components=num_com)
tr_pca2 = sk_pca2.fit_transform(tr_data)
te_pca2 = sk_pca2.transform(te_data)



clf = svm.SVC(random_state=42,C = 10,gamma = 0.001)
#clf = svm.SVC(random_state=42)
start_time = time.time()
clf.fit(tr_pca2,train_labels.values.ravel())
# p.fit(tr_pca2,train_labels.values.ravel())
end_time = time.time()
fit_time = end_time - start_time

print('fit time: ',fit_time)

s_time = time.time()
score=clf.score(te_pca2,test_labels)
e_time = time.time()
sc_time = e_time - s_time
print('score: ',score)
print('time to score ', sc_time)
print(clf.get_params())

