import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import torch
import pandas as pd
from shapelet_filter import ShapeletFilter
# from shapelet_classifier import ShapeletClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from candidate_shapelets import *
from shapelet_pruning import * 
from shapelet_coverage import *
from prune_and_coverage import *
from utils import *
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def shapelet_filter_test():
#     X_train, y_train = load_ucr_dataset_tensor('../data/Gun_Point/Gun_Point_TRAIN')
#     X_test, y_test = load_ucr_dataset_tensor('../data/Gun_Point/Gun_Point_TEST')
    
#     print("Using ShapeletFilter")
#     shapelet_filter = ShapeletFilter(min_length=13, max_length=75, k=15)
#     shapelet_filter.fit(X_train, y_train)
    
#     save_shapelets_to_file(shapelet_filter.shapelets, filename="shapelets_filter.xlsx", to_excel=True)
    
#     # Transform data
#     X_train_transformed = shapelet_filter.transform(X_train)
#     X_test_transformed = shapelet_filter.transform(X_test)
    
#     print("Transformed train shape:", X_train_transformed.shape)
#     print("Transformed test shape:", X_test_transformed.shape)
    
#     print("-------------------------------------")
    
#     print("Using ShapeletClassifier")
#     classifier = ShapeletClassifier(
#         min_length=13, 
#         max_length=75, 
#         k=15, 
#         classifier=RandomForestClassifier()
#     )
#     classifier.fit(X_train, y_train)
    
#     predictions = classifier.predict(X_test)
#     accuracy = classifier.score(X_test, y_test)
#     print("Predictions:", predictions)
#     print("Accuracy:", accuracy)
    
#     report = classification_report(y_test, predictions)
#     print("\nClassification Report:")
#     print(report)

def get_dynamic_threshold(length):
    if length <= 20:
        return 0.7
    elif length <= 40:
        return 1.0
    elif length <= 60:
        return 1.3
    else:
        return 1.6
    
def candidate_shapelets_test():
    # X_train, y_train = load_ucr_dataset_tensor('../data/Gun_Point/Gun_Point_TRAIN')
    # X_train, y_train = load_ucr_dataset_tensor('../data/ItalyPowerDemand/ItalyPowerDemand_TRAIN')
    X_train, y_train = load_ucr_dataset_tensor('../data/Coffee/Coffee_TRAIN')
    # X_train, y_train = load_ucr_dataset_tensor('../data/synthetic_control/synthetic_control_TRAIN')
    
    print("Data shape:", X_train.shape)
    print("Number of classes:", len(np.unique(y_train)))
    
    min_length = max(1, int(len(X_train) / 11))   # ensure integer >= 1
    max_length = int(len(X_train) / 2)
    
    print(f"Generating candidate shapelets with lengths from {min_length} to {max_length}...")
    
    # Use a subset of the data for faster testing
    subset_size = min(10, len(X_train))  # Use at most 10 time series
    X_subset = X_train[:subset_size]
    
    candidates = candidate_shapelets(X_subset, y_train, min_length, max_length)
    
    shapelets = []
    for shapelet in candidates[:5]:  # Lấy 5 candidate shapelet
        data = shapelet.data
        quality = shapelet.quality
        interval = pd.Interval(shapelet.start_pos, shapelet.start_pos + shapelet.length - 1, closed="both")
        series_id = shapelet.source_id
        start_pos = shapelet.start_pos
        length = shapelet.length
        class_label = shapelet.class_label

        shapelets.append(
            [data, quality, interval, series_id, start_pos, length, class_label]
        )

    print(shapelets)
    
    print(f"Generated {len(candidates)} candidate shapelets")
    
    save_shapelets_to_file(candidates, filename="candidate_shapelets_coffee.csv", to_excel=False)
    print("\nShapelets saved to files: candidate_shapelets_coffee.csv")

def prun_test():
    # df = pd.read_excel('result/candidate_shapelets.xlsx')
    df = pd.read_csv('candidate_shapelets_coffee.csv')
    df['values'] = df['values'].apply(ast.literal_eval)

    shapelets = []
    for _, row in df.iterrows():
        s = Shapelet(
            data=row['values'],
            start_pos=row['start_pos'],
            length=row['length'],
            source_id=row['source_id'],
            class_label=row.get('class_label', -1)
        )

        s.quality = row.get('quality', 0)
        
        # s.split_threshold = 0.6 * s.length  # ...% của độ dài shapelet
        s.split_threshold = get_dynamic_threshold(s.length) * s.length  # Dynamic threshold based on length
        shapelets.append(s)

    pruned_shapelets = ShapeletsPruning.prune(shapelets)

    save_shapelets(pruned_shapelets, filename="pruned_shapelets_coffee.csv", to_excel=False)
    print("\nShapelets saved to files: pruned_shapelets_coffee.csv")


def coverage_test(coverage_param=4):
    # df = pd.read_excel('pruned_shapelets_Coffee.xlsx')
    df = pd.read_csv('pruned_shapelets_coffee.csv')
    
    # df = pd.read_excel('result/pruned_shapelets.xlsx')
    df['values'] = df['values'].apply(ast.literal_eval)
        
    shapelets = []
    for _, row in df.iterrows():
        s = Shapelet(
            data=row['values'],
            start_pos=row['start_pos'],
            length=row['length'],
            source_id=row['source_id'],
            class_label=row['class_label']
        )
        
        try:
            s.quality = float(row['quality'])
            print(f"Read quality value: {s.quality} for shapelet from series {row['source_id']}")
        except (ValueError, KeyError) as e:
            print(f"Error reading quality: {e}")
            s.quality = 0
            
        s.covered_instances = [row['source_id']]
        shapelets.append(s)

    print(f"Applying Shapelet Coverage with coverage_param = {coverage_param}...")
    covered_shapelets = ShapeletsCoverage.cover(shapelets, coverage_param)

    print(f"\nSelected {len(covered_shapelets)} shapelets after coverage:")
    for i, s in enumerate(covered_shapelets):
        print(f"Shapelet {i+1}: Source ID={s.source_id}, Start={s.start_pos}, Len={s.length}, Class={s.class_label}, Quality={s.quality}")

    save_shapelets(covered_shapelets, filename="covered_shapelets_coffee.csv", to_excel=False)
    print("Saved to covered_shapelets_coffee.csv")


def prune_coverage():
    # df_candidate_shapelets = pd.read_excel('result(IG)/candidate_shapelets.xlsx')
    df_candidate_shapelets = pd.read_csv('covered_shapelets_coffee.csv')
    df_candidate_shapelets['values'] = df_candidate_shapelets['values'].apply(ast.literal_eval)
        
    # Convert DataFrame rows to Shapelet objects
    candidate_shapelets = []
    for _, row in df_candidate_shapelets.iterrows():
        s = Shapelet(
            data=row['values'],
            start_pos=row['start_pos'],
            length=row['length'],
            source_id=row['source_id'],
            class_label=row['class_label']
        )
        
        try:
            s.quality = float(row['quality'])
            # print(f"Read quality value: {s.quality} for shapelet from series {row['source_id']}")
        except (ValueError, KeyError) as e:
            print(f"Error reading quality: {e}")
            s.quality = 0
            
        # s.split_threshold = 0.6 * s.length
        s.split_threshold = get_dynamic_threshold(s.length) * s.length  # Dynamic threshold based on length
        
        # Initialize covered instances (at least covering its source instance)
        s.covered_instances = [row['source_id']]
        candidate_shapelets.append(s)
    
    coverage_param = 4
    
    # Test pruning
    print("\nAfter pruning:")
    pruned_shapelets = ShapeletsPruning.prune(candidate_shapelets)
    for i, shapelet in enumerate(pruned_shapelets):
        print(f"{i}: Class {shapelet.class_label}, Quality {shapelet.quality:.4f}, Length {shapelet.length}")
    
    # Test coverage
    print("\nAfter coverage (coverage param = 4):")
    covered_shapelets = ShapeletsCoverage.cover(pruned_shapelets, coverage_param)
    for i, shapelet in enumerate(covered_shapelets):
        print(f"{i}: Class {shapelet.class_label}, Quality {shapelet.quality:.4f}, Length {shapelet.length}")
    
    # Test PruneAndCoverage algorithm
    print("\nAfter PruneAndCoverage (coverage param = 4):")
    final_shapelets = PruneAndCoverage.prune_and_coverage(candidate_shapelets.copy(), coverage_param)
    for i, shapelet in enumerate(final_shapelets):
        print(f"{i}: Class {shapelet.class_label}, Quality {shapelet.quality:.4f}, Length {shapelet.length}")

    save_shapelets(final_shapelets, filename="pruned_covered_coffee.csv", to_excel=False)
    print("Saved to pruned_covered_coffee.csv")

        
def main():
    # print("Testing Shapelet Filter generation...")
    # shapelet_filter_test()

    print("Testing candidate shapelet generation...")
    candidate_shapelets_test()
    prun_test()
    coverage_test()
    prune_coverage()
    

if __name__ == "__main__":
    main()
    