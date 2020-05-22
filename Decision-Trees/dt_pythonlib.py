#Contributions: Minghong Xu, Zhixin Xie


# Load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
import math
import pandas as pd



txt_file = r"dt_data.txt"
df = pd.read_table(txt_file, skiprows = 0, sep = ', ', header = 0)

df = df.rename(columns={'(Occupied': 'Occupied'})
df = df.rename(columns={'Enjoy)': 'Enjoy'})
df.Enjoy = df.Enjoy.str.replace(';', '')
df.Occupied = df.Occupied.str.replace(r'^(\d*)[:]*[\s]*','')


df['Enjoy'],class_names = pd.factorize(df['Enjoy'])
print(class_names)



df['Occupied'],_ = pd.factorize(df['Occupied'])
df['Price'],_ = pd.factorize(df['Price'])
df['Music'],_ = pd.factorize(df['Music'])
df['Location'],_ = pd.factorize(df['Location'])
df['VIP'],_ = pd.factorize(df['VIP'])
df['Favorite Beer'],_ = pd.factorize(df['Favorite Beer'])
df



# Load data
X = df.drop('Enjoy', axis=1)
y = df['Enjoy']


# Create decision tree classifer object
clf = DecisionTreeClassifier(random_state=0)

# Train model
dtmodel = clf.fit(X, y)



#Load Test Data
test_data = {'Occupied':1, 'Price' : 2, 'Music' : 0, 'Location' : 1, 'VIP' : 1, 'Favorite Beer' : 1}
test_df = pd.DataFrame(test_data,index=[0]) 


print(dtmodel.predict(test_df))
print('[0:NO,1:YES]')
#[0:NO,1:YES]




#Graph
import graphviz
feature_names = X.columns

dot_data = tree.export_graphviz(dtmodel, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=class_names)
graph = graphviz.Source(dot_data)  
graph
