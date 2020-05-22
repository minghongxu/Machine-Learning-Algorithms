import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time


# gray scale image SVM


data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
#data[data>0]=data.applymap(lambda x: float(x/255))
#data: 42000,784(28*28)
# print(data.shape)
# print(data.columns)
# print(data[label==0])

# draw grey scale img
# for i in range(10):
#     train_ = data[label ==i]
#     data_new = []
#     for j in train_.index:
#         val = train_.loc[j].values.reshape(28,28)
#         data_new.append(val)
#     plt.figure(figsize=(25,25))
#     for h in range(1,5):
#         ax = plt.subplot(1,20,h)
#         ax.imshow(data_new[h],cmap='gray')

#split data
train, test,train_labels, test_labels = train_test_split(data, label, train_size=0.8, random_state=42)

clf = svm.SVC(random_state=42)
start_time = time.time()
clf.fit(train,train_labels.values.ravel())
end_time = time.time()
fit_time = end_time - start_time

print('fit time: ',fit_time)


s_time = time.time()
score=clf.score(test,test_labels)
e_time = time.time()
sc_time = e_time - s_time
print('score: ',score)
print('time to score ', sc_time)
print(clf.get_params())






