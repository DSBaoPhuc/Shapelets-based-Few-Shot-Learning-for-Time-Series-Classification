import numpy as np
from shapelet import *
from utils import subdist


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
    if len(orderline) == 0:
        return 0
    
    # Extract class labels
    classes = np.unique(y)
    n = len(orderline)
    
    # Calculate entropy
    class_counts = np.array([np.sum(y == c) for c in classes])
    class_probabilities = class_counts / n
    entropy_total = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
    
    best_ig = -float('inf')
    
    # Try different split points along the orderline
    for i in range(1, n):
        left = orderline[:i]
        right = orderline[i:]
        
        n_left = len(left)
        n_right = len(right)
        
        # Extract class labels for left and right
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
        
        # Calculate IG
        ig = entropy_total - ((n_left / n) * entropy_left + (n_right / n) * entropy_right)
        
        # print(f"Split at {i}: IG = {ig:.4f}, left_labels = {left_labels}, right_labels = {right_labels}")
        print(f"Split at {i}: IG = {ig:.4f}")
        
        # Update best IG
        if ig > best_ig:
            best_ig = ig
    
    return best_ig

def candidate_shapelets(T, y, min_length, max_length):
    """
    Algorithm 4: Generate candidate shapelets from time series data
    
    Parameters:
    -----------
    T : list of numpy arrays
        Time series data
    y : list or numpy array
        Class labels corresponding to the time series data
    min_length : int
        Minimum shapelet length
    max_length : int
        Maximum shapelet length
        
    Returns:
    --------
    candidate_shapelets : list
        List of candidate shapelets sorted by quality
    """
    print("Label distribution:", np.unique(y, return_counts=True))

    candidate_shapelets = []
    
    for i in range(len(T)):  # time series in T
        shapelets = []
        
        for l in range(min_length, min(max_length + 1, len(T[i]) + 1)):  # possible length
            for u in range(len(T[i]) - l + 1):  # start position
                S = T[i][u:u+l]  # Extract subsequence
                
                # Compute distances from time series to S
                distances = []
                orderline = []
                for m in range(len(T)):
                    dist = subdist(S, T[m])
                    distances.append(dist)
                    orderline.append((dist, y[m]))
                
                orderline.sort(key=lambda x: x[0])
                
                quality = assess_candidate(S, orderline, distances, y)
                
                print(f"Shapelet from series {i}, class {y[i]}, len={l}, pos={u}, quality={quality:.4f}")
                
                shapelet = Shapelet(S, u, l, i, y[i])
                shapelet.quality = quality
                shapelets.append(shapelet)
        
        # Sort shapelets by quality
        shapelets.sort(key=lambda x: x.quality, reverse=True)
        
        shapelets = remove_self_similar(shapelets)
        
        candidate_shapelets.extend(shapelets)
    
    return candidate_shapelets

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
    
    shapelets.sort(key=lambda x: x.quality, reverse=True)
    
    filtered = [shapelets[0]]
    
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
    if shapelet1.source_id != shapelet2.source_id:  # Use source_id instead of series_id
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

