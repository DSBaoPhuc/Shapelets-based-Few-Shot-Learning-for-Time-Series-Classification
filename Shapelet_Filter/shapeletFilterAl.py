import numpy as np
from math import log2
from collections import namedtuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Fix the namedtuple definition to match the parameters being used
Shapelet = namedtuple("Shapelet", ["subsequence", "start_pos", "length", "source_id", "class_label", "quality"])

def load_ucr_dataset_tensor(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            labels.append(parts[0])
            data.append(parts[1:])
    
    # Return NumPy arrays for compatibility with shapelet filter
    return np.array(data, dtype=np.float32), np.array(LabelEncoder().fit_transform(labels), dtype=np.int64)


def save_shapelets_to_file(shapelets, filename="shapelets.csv", to_excel=False):
    records = []
    for shapelet in shapelets:
        records.append({
            "source_id": shapelet.source_id,
            "start_pos": shapelet.start_pos,
            "length": shapelet.length,
            "class_label": shapelet.class_label,
            "quality": shapelet.quality,
            "values": shapelet.subsequence.tolist()
        })
    
    df = pd.DataFrame(records)
    
    if to_excel:
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)
        
        
def subdist(x, y):
    """
    Algorithm 3: Calculate minimum normalized distance between time series x and y
    
    Parameters:
    -----------
    x : numpy array
        First time series (shapelet)
    y : numpy array
        Second time series
        
    Returns:
    --------
    float
        Normalized distance between x and y
    """
    # Check if x is shorter than y
    if len(x) > len(y):
        return float('inf')
    
    best_sum = float('inf')  # MAX_VALUE
    
    # Normalize x
    x = z_norm(x)
    
    # For each possible starting position in y
    for i in range(len(y) - len(x) + 1):
        sum_dist = 0
        
        # Extract and normalize subsequence from y
        z = z_norm(y[i:i+len(x)])
        
        # Compute Euclidean Distance
        for j in range(len(x)):
            sum_dist += (z[j] - x[j])**2
        
        # Update best distance if current is better
        best_sum = min(best_sum, sum_dist)
    
    # Return normalized distance
    return np.sqrt(best_sum / len(x))


def z_norm(series):
    """
    Z-normalize a time series
    
    Parameters:
    -----------
    series : numpy array
        Time series to normalize
        
    Returns:
    --------
    numpy array
        Z-normalized time series
    """
    if len(series) == 0:
        return series
        
    mean = np.mean(series)
    std = np.std(series)
    
    # If standard deviation is zero, return zeros
    if std == 0:
        return np.zeros_like(series)
    
    return (series - mean) / std


def assess_candidate(S, orderline, distances, y):
    """
    Assess the quality of a candidate shapelet using Information Gain
    
    Parameters:
    -----------
    S : numpy array
        Shapelet (subsequence)
    orderline : list
        Sorted list of (distance, class_label) tuples
    distances : list
        Original distances list
    y : numpy array
        Class labels for the dataset
        
    Returns:
    --------
    quality : float
        Information Gain score of the shapelet
    """
    # If orderline is empty, return 0
    if len(orderline) == 0:
        return 0
    
    # Extract class labels from the dataset
    classes = np.unique(y)
    n = len(orderline)
    
    # Calculate entropy of the entire dataset
    class_counts = np.array([np.sum(y == c) for c in classes])
    class_probabilities = class_counts / n
    entropy_total = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
    
    best_ig = -float('inf')
    
    # Try different split points along the orderline
    for i in range(1, n):
        # Split the dataset
        left = orderline[:i]
        right = orderline[i:]
        
        n_left = len(left)
        n_right = len(right)
        
        # Extract class labels for left and right splits
        left_labels = [item[1] for item in left]
        right_labels = [item[1] for item in right]
        
        # Calculate entropy for left split
        left_counts = np.array([left_labels.count(c) for c in classes])
        left_probs = left_counts / n_left
        entropy_left = -np.sum(left_probs * np.log2(left_probs + 1e-10))
        
        # Calculate entropy for right split
        right_counts = np.array([right_labels.count(c) for c in classes])
        right_probs = right_counts / n_right
        entropy_right = -np.sum(right_probs * np.log2(right_probs + 1e-10))
        
        # Calculate information gain
        ig = entropy_total - ((n_left / n) * entropy_left + (n_right / n) * entropy_right)
        
        # Update best information gain
        if ig > best_ig:
            best_ig = ig
    
    return best_ig

def shapelet_filter(T, y, min_length, max_length, k):
    """
    Algorithm 1: Extract the top k shapelets from time series dataset T
    
    Parameters:
    -----------
    T : list of numpy arrays
        Time series dataset
    y : list or numpy array
        Class labels corresponding to the time series data
    min_length : int
        Minimum shapelet length
    max_length : int
        Maximum shapelet length
    k : int
        Number of shapelets to select
        
    Returns:
    --------
    kShapelets : list
        List of the top k shapelets
    """
    # Initialize empty list to hold k best shapelets
    kShapelets = []
    
    # For each time series in the dataset
    for i in range(len(T)):
        # Initialize empty list to hold shapelets from current series
        shapelets = []
        
        # For every possible length between min and max
        for l in range(min_length, min(max_length + 1, len(T[i]) + 1)):
            # For every possible starting position
            for u in range(len(T[i]) - l + 1):
                # Extract the subsequence
                S = T[i][u:u+l]
                
                # Compute distances from S to all time series
                distances = []
                orderline = []
                for m in range(len(T)):
                    dist = subdist(S, T[m])
                    distances.append(dist)
                    orderline.append((dist, y[m]))  # Store distance with class label
                
                # Sort orderline by distance
                orderline.sort(key=lambda x: x[0])
                
                # Assess candidate quality using information gain
                quality = assess_candidate(S, orderline, distances, y)
                
                # Create shapelet object - include quality directly in constructor
                shapelet = Shapelet(subsequence=S, start_pos=u, length=l, source_id=i, class_label=y[i], quality=quality)
                shapelets.append(shapelet)
        
        # Sort shapelets by quality (descending order)
        shapelets.sort(key=lambda x: x.quality, reverse=True)
        
        # Remove self-similar shapelets
        unique_shapelets = remove_self_similar(shapelets)
        
        # Merge with k best shapelets
        kShapelets.extend(unique_shapelets)
        
        # Keep only the k best shapelets overall
        kShapelets.sort(key=lambda x: x.quality, reverse=True)
        kShapelets = kShapelets[:k]
    
    return kShapelets

def remove_self_similar(shapelets):
    """
    Remove self-similar shapelets
    
    Parameters:
    -----------
    shapelets : list
        List of shapelets
        
    Returns:
    --------
    filtered_shapelets : list
        List with self-similar shapelets removed
    """
    if not shapelets:
        return []
    
    # Sort by quality if not already sorted
    shapelets.sort(key=lambda x: x.quality, reverse=True)
    
    filtered = [shapelets[0]]  # Keep the highest quality shapelet
    
    for i in range(1, len(shapelets)):
        is_self_similar = False
        
        for j in range(len(filtered)):
            # Use source_id instead of series_id
            if (shapelets[i].source_id == filtered[j].source_id and 
                are_overlapping(shapelets[i], filtered[j])):
                is_self_similar = True
                break
        
        if not is_self_similar:
            filtered.append(shapelets[i])
    
    return filtered


def are_overlapping(shapelet1, shapelet2):
    """
    Check if two shapelets overlap significantly
    
    Parameters:
    -----------
    shapelet1, shapelet2 : Shapelet objects
        The shapelets to compare
        
    Returns:
    --------
    bool
        True if shapelets overlap significantly
    """
    # Check if from same series
    if shapelet1.source_id != shapelet2.source_id:
        return False
    
    # Check if they overlap
    start1, end1 = shapelet1.start_pos, shapelet1.start_pos + shapelet1.length
    start2, end2 = shapelet2.start_pos, shapelet2.start_pos + shapelet2.length
    
    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    
    # Calculate overlap ratio (relative to the shorter shapelet)
    shorter_length = min(shapelet1.length, shapelet2.length)
    overlap_ratio = overlap_length / shorter_length
    
    # Return True if overlap is significant (e.g., more than 50%)
    return overlap_ratio > 0.5

def transform_dataset(shapelets, T):
    """
    Algorithm 2: Transform a dataset using extracted shapelets
    
    Parameters:
    -----------
    shapelets : list
        List of shapelet objects
    T : list of numpy arrays
        Time series dataset to transform
        
    Returns:
    --------
    transformed_dataset : numpy array
        Dataset transformed into shapelet space
    """
    transformed_dataset = []
    
    # For each time series in the dataset
    for i in range(len(T)):
        transformed_series = []
        
        # For each shapelet
        for j in range(len(shapelets)):
            # Calculate the distance between the time series and the shapelet
            dist = subdist(shapelets[j].subsequence, T[i])  # Use subsequence instead of data
            transformed_series.append(dist)
        
        transformed_dataset.append(transformed_series)
    
    return np.array(transformed_dataset)