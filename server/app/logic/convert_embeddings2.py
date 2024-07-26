from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

# For hyperparameter tuning of eps and min_samples
def tune_dbscan(X, eps_range, min_samples_range):
  best_score = -1
  best_params = []
  for eps in eps_range:
    for min_samples in min_samples_range:
      clustering = DBSCAN(eps=eps, min_samples=min_samples)
      labels = clustering.fit_predict(X)
      if len(set(labels)) > 1:  # Avoid single-cluster solutions
        score = silhouette_score(X, labels)
        if score > best_score:
          best_score = score
          best_params = [eps, min_samples]
  return best_params
    
def clustering1(list1):
    # Convert list1 to a numpy array if it's not already
    X = np.array(list1)
    eps_range = np.arange(10, 50, 5)
    min_samples_range = range(10, 50)
    tune_dbscan(X, eps_range, min_samples_range)
    
    # DBSCAN parameters
    dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1], metric='cosine').fit(X)
    
    arr = dbscan.labels_
    unique_values = np.unique(arr)
    
    indices_list = []
    for val in unique_values:
        indices = np.where(arr == val)[0]
        indices_list.append(indices)
    
    return indices_list
