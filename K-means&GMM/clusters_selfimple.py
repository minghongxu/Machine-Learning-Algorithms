#!/usr/bin/env python
# coding: utf-8



import math
import numpy as np
import random
from numpy import linalg as LA




txt_file = "clusters.txt"

a = np.loadtxt(txt_file, delimiter=',')


# ##### K-means


def distance(point1, point2):
    dis = ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
    return dis



def assign_centroid(array, centroid1, centroid2, centroid3):
    group1 = []
    group2 = []
    group3 = []
    for point in array:
        min_dis = min(distance(point,centroid1), distance(point,centroid2), distance(point,centroid3))
        if min_dis == distance(point,centroid1):
            group1.append(point)
        elif min_dis == distance(point,centroid2):
            group2.append(point)
        else:
            group3.append(point)
    return group1, group2, group3




def compute_centroid(group1, group2, group3):
    centroid1 = np.array([0.0,0.0])
    centroid2 = np.array([0.0,0.0])
    centroid3 = np.array([0.0,0.0])
    for point in group1:
        centroid1 += point
    for point in group2:
        centroid2 += point
    for point in group3:
        centroid3 += point
    return centroid1/len(group1), centroid2/len(group2), centroid3/len(group3)




def k_means(array, tolerance, max_iter):
    iter = 0
    x_max = max(array[: , 0])
    x_min = min(array[: , 0])
    y_max = max(array[: , 1])
    y_min = min(array[: , 1])
    centroid1 = [random.uniform(x_max,x_min),random.uniform(y_max,y_min)]
    centroid2 = [random.uniform(x_max,x_min),random.uniform(y_max,y_min)]
    centroid3 = [random.uniform(x_max,x_min),random.uniform(y_max,y_min)]
    while iter < max_iter:
        iter += 1
        group1, group2, group3 = assign_centroid(array, centroid1, centroid2, centroid3)
        centroid1_temp, centroid2_temp, centroid3_temp = compute_centroid(group1, group2, group3)
        if distance(centroid1,centroid1_temp) < tolerance and distance(centroid2,centroid2_temp) < tolerance and distance(centroid3,centroid3_temp) < tolerance:
            break
        centroid1, centroid2, centroid3 = centroid1_temp, centroid2_temp, centroid3_temp
    return centroid1, centroid2, centroid3




def weight(array, centroid1, centroid2, centroid3):
    total = 0
    group1, group2, group3 = assign_centroid(array, centroid1, centroid2, centroid3)
    for point in group1:
        total += distance(point, centroid1)
    for point in group2:
        total += distance(point, centroid2)
    for point in group3:
        total += distance(point, centroid3)
    return total





value = 100000

for i in range(20):
    centroid1, centroid2, centroid3 = k_means(a, 0.0001, 3000)
    v = weight(a, centroid1, centroid2, centroid3)
    if v < value:
        value = v
        final1, final2, final3  = centroid1, centroid2, centroid3
print(final1, final2, final3)


# ##### GMM



def get_mean(array, p):
    #p should be a row of probability
    s = np.array([0.0,0.0])
    for i in range(len(array)):
        s += array[i]*p[i]
    return s/sum(p)




def covarians_matrix(array, mean, p):
    #input mean here is flat array, not matrix
    d = len(mean)
    matrix = np.zeros([d,d])
    for i in range(len(array)):
        xi = (array[i]-mean).reshape(2,-1)
        matrix += np.dot(xi, xi.T)*p[i]
    return matrix/sum(p)




def multivariables_gaussian(x, mu, covarians_matrix):
    #input x,mu here should be flat array, not matrix
    x = x.reshape(2,-1)
    mu = mu.reshape(2,-1)
    d = len(x)
    pi, e = math.pi, math.e
    power = np.dot(-0.5*(x-mu).T, LA.inv(covarians_matrix))
    power = np.dot(power, x-mu)
    p = (1/(pi**(d/2))) * (LA.det(covarians_matrix)**-0.5) * (e**power)
    return p[0][0]








def normalize(x):
    sum = 0
    for value in x:
        sum += value
    return x/sum



def find_parameters(array, ric):
    #output elements in list mu is flat array, not matrix
    mu = []
    cov_matrices = []
    amp = []
    for row in ric:
        array_temp = np.zeros([array.shape[0],array.shape[1]])
        sum_r = 0
        for i in range(len(row)):
            array_temp[i] = array[i]*row[i]
            sum_r += row[i]
        mean = get_mean(array,row)
        mu.append(mean)
        cov_matrices.append(covarians_matrix(array, mean, row))
        amp.append(sum_r/len(array))#need make sure
    return mu, cov_matrices, amp




def find_ric(array, mu, cov_matrices, amp):
    k = len(mu)
    ric = np.zeros([k,len(array)])
    for c in range(k):
        for i in range(len(array)):
            denominator = 0
            for j in range(k):  
                denominator += amp[j]*multivariables_gaussian(array[i], mu[j], cov_matrices[j])
            ric[c,i] =  (amp[c]*multivariables_gaussian(array[i], mu[c], cov_matrices[c]))/denominator
    return ric




#learn more later
def log_likelihood(ric):
    s = 0
    result = 0
    for row in ric:
        for value in row:
            s += value
            result += value * math.log(value)
    return s*math.log(s) - result




def max_dif(ric, ricnew):
    dif = 0
    for i in range(ric.shape[0]):
        for j in range(ric.shape[1]):
            diftemp = abs(ricnew[i,j]-ric[i,j])
            if diftemp > dif:
                dif = diftemp
    return dif




def GMM(array, k, tolerance, max_iter):
    iter = 0
    ric = np.random.uniform(0,1, size=(k,len(array)))
    for i in range(len(array)):
        ric[:,i] = normalize(ric[:,i])
    mu = []
    cov_matrices = []
    amp = []
    while iter < max_iter:
        iter += 1
        mu, cov_matrices, amp = find_parameters(array, ric)
        ric_temp = find_ric(array, mu, cov_matrices, amp)
        if max_dif(ric, ric_temp) < tolerance:
            break
        ric = ric_temp
    return mu, cov_matrices, amp




log = 0
for i in range(20):
    mu, cov_matrices, amp = GMM(a, 3, 0.0001, 500)
    ric = find_ric(a, mu, cov_matrices, amp)
    if log_likelihood(ric) > log:
        log = log_likelihood(ric)
        final_mu, final_cov_matrices, final_amp = mu, cov_matrices, amp
print(final_mu, final_cov_matrices, final_amp)

