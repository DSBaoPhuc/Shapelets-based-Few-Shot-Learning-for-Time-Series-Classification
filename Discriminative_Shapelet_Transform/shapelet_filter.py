import numpy as np
from collections import defaultdict
from shapelet import Shapelet
from utils import subdist

class ShapeletFilter:
    """
    Implementation of the shapelet filter algorithm for time series classification
    """
    def __init__(self, min_length=13, max_length=None, k=10):
        self.min_length = min_length
        self.max_length = max_length
        self.k = k
        self.shapelets = []
    
    def fit(self, X, y):
        """
        Extract k best shapelets from the training data
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_timesteps)
            Time series data
        y : numpy array of shape (n_samples,)
            Class labels
        """
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy()
        if hasattr(y, 'numpy'):
            y = y.cpu().numpy()
            
        if self.max_length is None:
            self.max_length = X.shape[1] // 2
        
        X_list = [X[i, :] for i in range(X.shape[0])]
        y_list = y.tolist() if hasattr(y, 'tolist') else list(y)
        
        k_shapelets = self._shapelet_filter(X_list, y_list, self.min_length, self.max_length, self.k)
        self.shapelets = k_shapelets
        return self
    
    def transform(self, X):
        """
        Transform time series data using the extracted shapelets
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_timesteps)
            Time series data
        
        Returns:
        --------
        X_transformed : numpy array
            Transformed data where each column is the distance to a shapelet
        """
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy()
            
        X_list = [X[i, :] for i in range(X.shape[0])]
        
        return self._transform_data(self.shapelets, X_list)
    
    def _shapelet_filter(self, T, y, min_length, max_length, k):
        """
        Algorithm 1: Extract k best shapelets from the time series data
        """
        k_shapelets = []
        
        for i in range(len(T)):
            shapelet_set = []
            
            for l in range(min_length, min(max_length + 1, len(T[i]) + 1)):
                for u in range(len(T[i]) - l + 1):
                    S = T[i][u:u+l]
                    
                    distances = []
                    for m in range(len(T)):
                        dist = subdist(S, T[m])
                        distances.append((dist, y[m]))
                    
                    ordered_distances = sorted(distances, key=lambda x: x[0])
                    quality = self._assess_candidate(S, ordered_distances, distances)
                    
                    shapelet = Shapelet(S, u, l, i, y[i])
                    shapelet.quality = quality
                    shapelet_set.append(shapelet)
            
            shapelet_set = sorted(shapelet_set, key=lambda x: x.quality, reverse=True)
            shapelet_set = self._remove_self_similar(shapelet_set)
            k_shapelets = self._merge(k, k_shapelets, shapelet_set)
        
        return k_shapelets
    
    def _transform_data(self, shapelets, dataset):
        """
        Algorithm 2: Transform time series data using the extracted shapelets
        """
        output = []
        
        for i in range(len(dataset)):
            transformed = []
            for shapelet in shapelets:
                dist = subdist(shapelet.data, dataset[i])
                transformed.append(dist)
            output.append(transformed)
        
        return np.array(output)
    
    def _assess_candidate(self, shapelet, orderline, distances):
        """
        Assess the quality of a shapelet candidate
        This could be information gain, F-stat, etc.
        """
        class_distribution = defaultdict(list)
        for dist, label in distances:
            class_distribution[label].append(dist)
        
        # If only one class, return 0
        if len(class_distribution) <= 1:
            return 0
        
        # Calculate separation between classes
        means = {label: np.mean(dists) for label, dists in class_distribution.items()}
        overall_mean = np.mean([dist for dist, _ in distances])
        
        between_group_variance = sum([len(dists) * (means[label] - overall_mean) ** 2 
                                    for label, dists in class_distribution.items()])
        
        within_group_variance = sum([sum([(d - means[label]) ** 2 for d in dists])
                                   for label, dists in class_distribution.items()])
        
        # F-statistic inspired quality measure
        if within_group_variance == 0:
            return float('inf')
        
        return between_group_variance / within_group_variance
    
    def _remove_self_similar(self, shapelets):
        """
        Remove self-similar shapelets from the same time series
        """
        if not shapelets:
            return []
        
        filtered = [shapelets[0]]
        
        for i in range(1, len(shapelets)):
            similar = False
            for j in range(len(filtered)):
                # Check if they're from the same series
                if shapelets[i].source_id == filtered[j].source_id:
                    # Check if they overlap
                    s1_start, s1_end = shapelets[i].start_pos, shapelets[i].start_pos + shapelets[i].length
                    s2_start, s2_end = filtered[j].start_pos, filtered[j].start_pos + filtered[j].length
                    
                    # If they overlap and the new one has lower quality, it's similar
                    if not (s1_end <= s2_start or s1_start >= s2_end) and shapelets[i].quality < filtered[j].quality:
                        similar = True
                        break
            
            if not similar:
                filtered.append(shapelets[i])
        
        return filtered
    
    def _merge(self, k, best_so_far, new_shapelets):
        """
        Merge k best shapelets with new candidates and return top k
        """
        merged = best_so_far + new_shapelets
        merged.sort(key=lambda x: x.quality, reverse=True)
        return merged[:k]