import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from shapeletFilterAl import shapelet_filter, transform_dataset, load_ucr_dataset_tensor
import time
import pandas as pd

def run_shapelet_pipeline(train_path, test_path, min_length, max_length, k):
    """
    Run the complete shapelet pipeline:
    - Extract shapelets from training data
    - Transform both training and test data
    - Train a classifier and evaluate performance
    
    Parameters:
    -----------
    train_path : str
        Path to training data
    test_path : str
        Path to test data
    min_length : int
        Minimum shapelet length
    max_length : int
        Maximum shapelet length
    k : int
        Number of shapelets to extract
    """
    
    # Load datasets
    print("Loading datasets...")
    X_train, y_train = load_ucr_dataset_tensor(train_path)
    X_test, y_test = load_ucr_dataset_tensor(test_path)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Extract shapelets
    print(f"\nExtracting top {k} shapelets with lengths {min_length}-{max_length}...")
    start_time = time.time()
    shapelets = shapelet_filter(X_train, y_train, min_length, max_length, k)
    extraction_time = time.time() - start_time
    print(f"Shapelet extraction completed in {extraction_time:.2f} seconds")
    
    # Display top shapelets
    print(f"\nTop {min(5, len(shapelets))} shapelets:")
    for i, shapelet in enumerate(shapelets[:5]):
        print(f"Shapelet {i+1}:")
        print(f"  Series ID: {shapelet.source_id}")
        print(f"  Start position: {shapelet.start_pos}")
        print(f"  Length: {shapelet.length}")
        print(f"  Class label: {shapelet.class_label}")
        print(f"  Quality: {shapelet.quality:.4f}")
    
    # Transform datasets
    print("\nTransforming datasets using shapelets...")
    start_time = time.time()
    X_train_transformed = transform_dataset(shapelets, X_train)
    X_test_transformed = transform_dataset(shapelets, X_test)
    transform_time = time.time() - start_time
    print(f"Transformation completed in {transform_time:.2f} seconds")
    
    print(f"Transformed training data shape: {X_train_transformed.shape}")
    print(f"Transformed testing data shape: {X_test_transformed.shape}")
    
    # Train and evaluate classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_transformed, y_train)
    
    # Make predictions and evaluate
    y_pred = clf.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Visualize shapelets
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(shapelets))):
        plt.subplot(1, 5, i+1)
        plt.plot(shapelets[i].data)
        plt.title(f"Shapelet {i+1}\nQuality: {shapelets[i].quality:.4f}")
    plt.tight_layout()
    plt.savefig("top_shapelets.png")
    plt.show()
    
    # Save shapelets
    save_shapelets_to_file(shapelets, "extracted_shapelets.xlsx", to_excel=True)
    print("\nShapelets saved to extracted_shapelets.xlsx and visualization saved to top_shapelets.png")

def save_shapelets_to_file(shapelets, filename="shapelets.xlsx", to_excel=True):
    """
    Save shapelets to a file (Excel or CSV)
    
    Parameters:
    -----------
    shapelets : list
        List of shapelet objects
    filename : str
        Name of output file
    to_excel : bool
        Whether to save as Excel (True) or CSV (False)
    """
    
    data = []
    for i, shapelet in enumerate(shapelets):
        row = {
            "Id": i + 1,
            "Series_ID": shapelet.source_id,
            "Start": shapelet.start_pos,
            "Length": shapelet.length,
            "Class": shapelet.class_label,
            "Quality": shapelet.quality
        }
        
        # Add shapelet values
        for j, val in enumerate(shapelet.data):
            row[f"Value_{j+1}"] = val
            
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if to_excel:
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)

# Example usage
if __name__ == "__main__":
    # Set parameters
    train_path = "../data/Gun_Point/Gun_Point_TRAIN"
    test_path = "../data/Gun_Point/Gun_Point_TEST"
    min_length = 13
    max_length = 75
    k = 20
    
    # Run pipeline
    run_shapelet_pipeline(train_path, test_path, min_length, max_length, k)