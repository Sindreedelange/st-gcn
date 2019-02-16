import numpy as np

def normalize(arr):
    min_val = np.amin(arr)
    max_val = np.amax(arr)
    diff = max_val - min_val
    return ((arr - min_val) / diff) 

def round_traditional(val, digits):
    return round(val+10**(-len(str(val))-1), digits)