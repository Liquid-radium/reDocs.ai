from sklearn.cluster import DBSCAN
import numpy as np

def clustering1(list1):
    # Convert list1 to a numpy array if it's not already
    X = np.array(list1)
    
    # DBSCAN parameters
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine').fit(X)
    
    arr = dbscan.labels_
    unique_values = np.unique(arr)
    
    indices_list = []
    for val in unique_values:
        indices = np.where(arr == val)[0]
        indices_list.append(indices)
    
    # For debugging or inspection
    # for i in range(len(indices_list)):
    #     print("Cluster {}: {}".format(i, list(indices_list[i])))
    
    return indices_list
