import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# file_path = './result(IG)/covered_shapelets.xlsx'
# file_path = './result(IG)/pruned_shapelets.xlsx'
file_path = './result(IG)/pruned_covered.xlsx'

data = pd.read_excel(file_path)

print("First few rows of data:")
print(data.head())

if isinstance(data['values'].iloc[0], str):
    try:
        data['values'] = data['values'].apply(lambda x: np.array([float(v) for v in x.strip('[]').split(',')]))
    except Exception as e:
        print(f"Error parsing values: {e}")
        print(f"Problematic value example: {data['values'].iloc[0]}")
    
    # Convert values column to a set of feature columns
    values_expanded = pd.DataFrame(data['values'].tolist())
    
    X = values_expanded
else:
    X = pd.DataFrame(data['values'].tolist())

nan_count = X.isna().sum().sum()
print(f"Number of NaN values in features: {nan_count}")

print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

y = data['class_label']

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

print(f"Data shape: {data.shape}")
print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")

print(f"Training target distribution: {pd.Series(y_train).value_counts()}")
print(f"Testing target distribution: {pd.Series(y_test).value_counts()}")

classifiers = {
    "1-NN": KNeighborsClassifier(n_neighbors=1),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Naive Bayes": GaussianNB()
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("---------------------------------")
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(f"First 5 predictions vs actual: {list(zip(y_pred[:5], y_test.iloc[:5]))}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"{name} Confusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()