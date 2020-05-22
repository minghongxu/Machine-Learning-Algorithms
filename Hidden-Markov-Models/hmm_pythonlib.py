#!/usr/bin/env python
# coding: utf-8

# In[5]:



import math
import numpy as np
from hmmlearn import hmm


# In[325]:


file= 'hmm-data.txt'


# In[326]:


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


# In[327]:


grid_world, noisy_dis = read_file(file)


# In[328]:


n_observations = len(noisy_dis)


# In[329]:


def get_free_grids(file):
    f = open(file, "r")
    count = 0
    free_grids = []
    for line in f:
        if count > 1 and count < 12:
            grid = line.split()
            for i in range(len(grid)):
                if grid[i] == "1":
                    free_grids.append((count-2, i))

        count += 1
    
    return free_grids


# In[330]:


free_grids = get_free_grids(file)
n_states = len(free_grids)


# In[331]:


tower_loc = np.array([[0,0],[0,9],[9,0],[9,9]])


# In[332]:


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


# In[333]:


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


# In[334]:

distance_dict = find_distance_matrix(grid_world, tower_loc)
possible_cells = []
for i in range(0, len(noisy_dis)):
    cells = find_possible_cells(grid_world, noisy_dis[i], distance_dict)
    possible_cells.append(cells)


# In[335]:


def is_neighbor(cell1, cell2):
    diff = np.array(cell1) - np.array(cell2)
    total = np.sum(diff**2)
    return total == 1


# In[336]:


def get_transit_probability(cell):
    count = 0
    x = cell[0]
    y = cell[1]
    if (x + 1, y) in free_grids:
        count += 1
    if (x - 1, y) in free_grids:
        count += 1
    if (x, y + 1) in free_grids:
        count += 1
    if (x, y - 1) in free_grids:
        count += 1
    return 1/count


# In[337]:


start_probability = [1/n_states for i in range(n_states)]
transition_probability = []
for i in range(n_states):
    trans = []
    for j in range(n_states):
        if is_neighbor(free_grids[i], free_grids[j]):
            trans.append(get_transit_probability(free_grids[i]))
        else:
            trans.append(0)
    transition_probability.append(trans)


# In[338]:


def get_emission_probability(node, tower_loc):
    probability = 1
    for tower in tower_loc:
        distance = np.sqrt((node[0]- tower[0])**2 + (node[1]- tower[1])**2)
        probability *= 1/(distance * 1.3 - distance * 0.7)
    return probability


# In[339]:


emission_probability = []
for i in range(n_states):
    emission = []
    for j in range(n_observations):
        if free_grids[i] in possible_cells[j]:
            emission.append(get_emission_probability(free_grids[i], tower_loc))
        else:
            emission.append(0)
    
    emission_probability.append(emission)


# In[340]:


model = hmm.MultinomialHMM(n_components=n_states) 
model.startprob_ = np.array(start_probability)   
model.transmat_ = np.array(transition_probability) 
model.emissionprob_ = np.array(emission_probability) 
seen_list = [i for i in range(n_observations)]
seen = np.array([seen_list]).T  
hmm_model = model.predict(seen) 




print("The most possible path is:")
print ( ", ".join(map(lambda x: str(free_grids[x]), hmm_model)))



