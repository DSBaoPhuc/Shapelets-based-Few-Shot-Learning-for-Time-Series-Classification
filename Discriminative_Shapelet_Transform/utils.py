import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_ucr_dataset_tensor(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            labels.append(parts[0])
            data.append(parts[1:])
    
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
            "values": shapelet.data.tolist()
        })
    
    df = pd.DataFrame(records)
    
    if to_excel:
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)
        
def save_shapelets(shapelets, filename="shapelets.csv", to_excel=False):
    records = []
    for shapelet in shapelets:
        records.append({
            "source_id": shapelet.source_id,
            "start_pos": shapelet.start_pos,
            "length": shapelet.length,
            "class_label": shapelet.class_label,
            "quality": shapelet.quality,
            "values": shapelet.data 
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
    if len(x) > len(y):
        return float('inf')
    
    best_sum = float('inf')  # MAX_VALUE
    
    x = z_norm(x)
    
    for i in range(len(y) - len(x) + 1):
        sum_dist = 0
        
        z = z_norm(y[i:i+len(x)])
        
        for j in range(len(x)):
            sum_dist += (z[j] - x[j])**2
        
        best_sum = min(best_sum, sum_dist)
    
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