import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA


# gray scale image + PCA + SVM


data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
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
plt.figure(figsize=(20,10))
ind = np.arange(len(perOfVariance))
plt.bar(ind,perOfVariance,color = 'green')
plt.xlabel('#components')
plt.ylabel('Variance')
plt.show()

num_com = len(perOfVc[perOfVc<=0.9])
print('chose 90% attributes:', num_com)
sk_pca2 = sklearnPCA(n_components=num_com)
tr_pca2 = sk_pca2.fit_transform(s_train)
te_pca2 = sk_pca2.transform(s_test)


clf = svm.SVC(random_state=42,C = 10,gamma = 0.001)
start_time = time.time()
clf.fit(tr_pca2,train_labels.values.ravel())
end_time = time.time()
fit_time = end_time - start_time

print('fit time: ',fit_time)

# predict = clf.predict(test)
# accuracy = metrics.accuracy_score(test,predict)
# print('accuracy',accuracy)

s_time = time.time()
score=clf.score(te_pca2,test_labels)
e_time = time.time()
sc_time = e_time - s_time
print('predict: ',score)
print('time to predict ', sc_time)
print(clf.get_params())


