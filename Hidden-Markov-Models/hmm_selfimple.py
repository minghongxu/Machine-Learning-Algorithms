#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import numpy as np


# In[3]:


file= 'hmm-data.txt'


# In[4]:


def read_file(file):
    grid_world = []
    noisy_dis = []
    with open(file) as f:
        count = 0
        for line in f:
            if count in range(2,12):
                grid_world.append(line.split())
            elif count in range(24,35):
                noisy_dis.append(line.split())
            count += 1
    return np.array(grid_world), np.array(noisy_dis)


# In[5]:


grid_world, noisy_dis = read_file(file)


# In[6]:


tower_loc = np.array([[0,0],[0,9],[9,0],[9,9]])


# In[ ]:





# In[7]:


def find_distance_matrix(grid_world, tower_loc):
    distance_dict = {}
    row,col = grid_world.shape
    for i in range(row):
        for j in range(col):
            if grid_world[i][j] == '1':
                four_dis = []
                for tower in tower_loc:
                    dis = ((j - tower[1])**2 + (i-tower[0])**2)**0.5
                    four_dis.append((0.7 * dis, 1.3 * dis))
                distance_dict[(i,j)] = four_dis
    return distance_dict


# In[ ]:





# In[8]:


def find_possible_cells(grid_world, noisy_dis, distance_dict):
    possible_cells = []
    row,col = grid_world.shape
    for i in range(row):
        for j in range(col):
            if grid_world[i][j] == '1':
                count = 0
                dis_range = distance_dict[(i,j)]
                for tower_num in range(4):
                    if float(noisy_dis[tower_num]) > dis_range[tower_num][0] and                     float(noisy_dis[tower_num]) < dis_range[tower_num][1]:
                        count += 1
                if count == 4:
                    possible_cells.append((i,j))
    return possible_cells


# In[ ]:





# In[9]:


def find_transition_probability(grid_world,cell):
    count = 0
    x, y = cell[0], cell[1]
    transition_probability = {}
    if x - 1 >= 0:
        if grid_world[x-1][y] == '1':
            count += 1
            transition_probability[(x-1, y)] = 0
    if y - 1 >= 0:
        if grid_world[x][y-1] == '1':
            count += 1
            transition_probability[(x, y-1)] = 0
    if x + 1 < 10:
        if grid_world[x+1][y] == '1':
            count += 1
            transition_probability[(x+1, y)] = 0
    if y + 1 < 10:
        if grid_world[x][y+1] == '1':
            count += 1
            transition_probability[(x, y+1)] = 0
    for key in transition_probability:
        transition_probability[key] = 1/count
    return transition_probability


# In[ ]:





# In[21]:


def find_evidence_probability(distance_dict,cell):
    range_list = distance_dict[cell]
    result = 1
    for pair in range_list:
        scope = (pair[1] - pair[0])/0.1
        if scope == 0:
            result = result * 1
        else:
            result = result * (1/scope)
    return result


# In[ ]:





# In[32]:


def viterbi(grid_world, tower_loc, noisy_dis, distance_dict):
    possible_path = {}
    possible_cells_zero = find_possible_cells(grid_world, noisy_dis[0], distance_dict)
    possible_path[0] = {}
    for cell in possible_cells_zero:
        possible_path[0][cell] = {}
        possible_path[0][cell]['previous'] = 'root'
        possible_path[0][cell]['prob'] = find_evidence_probability(distance_dict, cell)   
        
    for step in range(1, len(noisy_dis)):
        possible_path[step] = {}
        possible_cells = find_possible_cells(grid_world, noisy_dis[step], distance_dict)
        for cell in possible_path[step-1]:
            transition_probability = find_transition_probability(grid_world,cell)
            for neighbor in transition_probability:
                if neighbor in possible_cells:
                    prob = transition_probability[neighbor] * find_evidence_probability(distance_dict, cell)                    * possible_path[step-1][cell]['prob']
                    if neighbor in possible_path[step]:
                        possible_path[step][neighbor]['previous'][cell] = prob
                        if prob > possible_path[step][neighbor]['prob']:
                            possible_path[step][neighbor]['prob'] = prob
                    else:
                        possible_path[step][neighbor] = {}
                        possible_path[step][neighbor]['previous'] = {}
                        possible_path[step][neighbor]['previous'][cell] = prob
                        possible_path[step][neighbor]['prob'] = prob
    
    return possible_path
                        

        
    


# In[ ]:





# In[11]:


def max_key_in_dict(dic):
    k = 0
    value = -100000
    for key in dic:
        if dic[key] > value:
            value = dic[key]
            k = key
    return k


# In[12]:


def most_possible_paths(possible_path):
    max_prob = 0
    path = []
    p = 0
    for point in possible_path[10]:
        if possible_path[10][point]['prob'] > max_prob:
            max_prob = possible_path[10][point]['prob']
            p = point
    path.append(p)
    p = max_key_in_dict(possible_path[10][p]['previous'])
    path.append(p)
    for step in range(9,0,-1):
        for point in possible_path[step]:
            if point == p:
                p = max_key_in_dict(possible_path[step][p]['previous'])
                path.append(p)
    path_forward = []
    
    for i in range(len(path)):
        step = 10 - i
        path_forward.append(path[step])
    return path_forward
                


# In[ ]:





# In[ ]:





# In[25]:


distance_dict = find_distance_matrix(grid_world, tower_loc)


# In[33]:


possible_path = viterbi(grid_world, tower_loc, noisy_dis, distance_dict)


# In[106]:


print("The most possible path is:")


# In[35]:


print(most_possible_paths(possible_path))


# In[ ]:





# In[ ]:




