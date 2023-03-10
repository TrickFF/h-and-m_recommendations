import numpy as np

from numba import jit, typeof, typed, types, prange

     
@jit(nopython=True, fastmath=True)
def precision_at_k(recommended_list, bought_list, k=12):
    
    recommended_list = recommended_list[:k]
    
    flags = np.zeros(len(recommended_list))
    for i in range(len(recommended_list)):
        if recommended_list[i] in bought_list:
            flags[i] = 1
    
    precision = flags.sum() / len(recommended_list)
    
    return precision
    
@jit(nopython=True, fastmath=True)
def ap_k(recommended_list, bought_list, k):
    
    flags = np.zeros(len(recommended_list))
    for i in range(len(recommended_list)):
        if recommended_list[i] in bought_list:
            flags[i] = 1
    
    sum_ = 0
    
    for i in range(k):
        if flags[i] == 1:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            sum_ += p_k
            
    result = sum_ / k
    
    return result
    
@jit(nopython=True, fastmath=True)
def recall(recommended_list, bought_list):
    
    flags = np.zeros(len(bought_list))
    for i in range(len(bought_list)):
        if bought_list[i] in recommended_list:
            flags[i] = 1
    
    recall = flags.sum() / len(bought_list)
    
    return recall
    
@jit(nopython=True, fastmath=True)
def recall_at_k(recommended_list, bought_list, k):
        
    recommended_list = recommended_list[:k]
    
    flags = np.zeros(len(bought_list))
    for i in range(len(bought_list)):
        if bought_list[i] in recommended_list:
            flags[i] = 1
    
    recall = flags.sum() / len(bought_list)
    
    return recall