import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import time

# binary image + SVM

data = pd.read_csv('train.csv')
#print(data.shape)

#remove label
label = data.label
data = data.drop('label',axis = 1)
data[data>0]=1

# train = data
# train_label = label
# for i in range(0,10):
#     train_data = train[train_label == i]
#     new_data = []
#     for j in train_data.index:
#         value = train_data.loc[j].values.reshape(28,28)
#         new_data.append(value)
#     plt.figure(figsize=(25,25))
#     for h in range(1,5):
#         ax = plt.subplot(1,20,h)
#         ax.imshow(new_data[h],cmap = 'binary')

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
