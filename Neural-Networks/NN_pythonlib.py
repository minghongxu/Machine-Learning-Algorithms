#!/usr/bin/env python
# coding: utf-8

# Group Member: Minghong Xu, Zhixin Xie
# In[5]:


from sklearn.neural_network import MLPClassifier


# In[10]:


def load_pgm(pgm):
    with open(pgm, 'rb') as f:
        f.readline() 
        f.readline()   
        x, y = f.readline().split()
        x = int(x)
        y = int(y)
        max_scale = int(f.readline().strip())
        image = []
        for i in range(x * y):
            image.append((f.read(1)[0]) / max_scale)
        return image


# In[28]:


images = []
results = []

with open('downgesture_train.list') as f:
    for image in f.readlines():
        image = image.strip()
        images.append(load_pgm(image))
        if 'down' in image:
            results.append(1)
        else:
            results.append(0)

classifier = MLPClassifier(solver='sgd', alpha=0,
                  hidden_layer_sizes=(100,), activation='logistic', learning_rate_init=0.1,
                  max_iter=1000)

classifier.fit(images, results)


# In[48]:


correct = 0
total = 0
with open('downgesture_test.list') as f:
    for image in f.readlines():
        total += 1
        image = image.strip()
        p = classifier.predict([load_pgm(image),])[0]
        print(image,p)
        if (p == 1) == ('down' in image):
            correct += 1
accuracy = correct/total
print('accuracy:',accuracy)

