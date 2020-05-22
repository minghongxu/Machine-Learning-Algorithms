#Contributions: Minghong Xu, Zhixin Xie

import math
import pandas as pd




txt_file = r"dt_data.txt"
df = pd.read_table(txt_file, skiprows = 0, sep = ', ', header = 0)

df = df.rename(columns={'(Occupied': 'Occupied'})
df = df.rename(columns={'Enjoy)': 'Enjoy'})

df.Enjoy = df.Enjoy.str.replace(';', '')

df.Occupied = df.Occupied.str.replace(r'^(\d*)[:]*[\s]*','')
print(df.head())


def entropy(l:list):
    e = 0
    for p in l:
        if p == 0 or p==1:
            pass
        else:
            e = e-p*math.log(p,2)
    return e


def p_list(df, column):
    l = []
    for i in df[column].value_counts():
        l.append(i/len(df))
    return l


def information_gain(df, column, label):
    entropy_before = entropy(p_list(df, label))
    entropy_after = 0
    for i in df[column].value_counts().keys():
        df_temp = df.loc[df[column] == i]
        entropy_after += entropy(p_list(df_temp, label))*(len(df_temp)/len(df))
    return entropy_before - entropy_after


print(information_gain(df, 'Occupied', 'Enjoy'))



def choose_best_column(df, label):
    best_column = 0
    max_infor = 0
    for i in df.columns:
        if i == label:
            return best_column
        infor = information_gain(df, i, label)
        if infor > max_infor:
            best_column = i
            max_infor = infor
    return best_column


print(choose_best_column(df, 'Enjoy'))


def create_tree(df, label):
    tree = {}
    best_column = choose_best_column(df, label)
    tree[best_column] = {}
    for i in df[best_column].value_counts().keys():
        dftemp = df.loc[df[best_column] == i] 
        #print(information_gain(dftemp, sub_best_column, label))
        if entropy(p_list(dftemp, label)) == 0:
            tree[best_column].update({i:dftemp[label].values[0]})
        elif choose_best_column(dftemp, label) == 0:
            tree[best_column].update({i:dftemp[label].value_counts().keys()[0]})
        else:
            tree[best_column].update({i:create_tree(dftemp, label)})
    return tree



decision_tree = create_tree(df, 'Enjoy') 

test_data = {'Occupied':'Moderate', 'Price' : 'Cheap', 'Music' : 'Loud', 'Location' : 'City-Center', 'VIP' : 'No', 'Favorite Beer' : 'No'}

def predict(tree, tdata):
    key = list(tree.keys())[0]
    value = tdata[key]
    result = tree
    while type(result) is dict:
        result = result[key][value]
        if type(result) is not dict:
            return result
        key = list(result.keys())[0]
        value = tdata[key]
    return result

print(predict(decision_tree, test_data))
print(decision_tree)



#Below is for drawing
count = 10
def number_dict(d):
    global count
    l = list(d.keys())
    for i in l:
        if type(d[i]) is dict:
            number_dict(d[i])
        d[i+str(count)] = d.pop(i)
        count += 1


def tuple_list(tree):
    l = []
    for i in tree:
        for j in tree[i]:
            if type(tree[i][j]) is not dict:
                l.append((i,tree[i][j],j))
            else:
                for k in tree[i][j]:
                    l.append((i,k,j))
                l = l + tuple_list(tree[i][j])
    return l


from graphviz import Digraph

def draw_tree(tree):
    gz=Digraph("Decision Tree",'comment',None,None,'png',None,"UTF-8",
               {'rankdir':'TB'},
               {'color':'black','fontcolor':'black','fontname':'FangSong','fontsize':'12','style':'rounded','shape':'box'},
               {'color':'#999999','fontcolor':'#888888','fontsize':'10','fontname':'FangSong'},None,False)
    treecopy = tree.copy()
    number_dict(treecopy)
    l = tuple_list(treecopy)
    for t in l:
        gz.node(t[0],t[0])
        gz.node(t[1],t[1])
        gz.edge(t[0],t[1],t[2])
    return gz




draw_tree(decision_tree)

