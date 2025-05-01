import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def train_gesture_classifier(data_dir="gesture_data"):
    print("Loading training data...")
    
    # Load the data
    X = []
    y = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please run data_collection.py first to collect gesture samples.")
        return None
    
    # Count total files to process
    total_files = 0
    for gesture_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, gesture_class)
        if os.path.isdir(class_dir):
            total_files += len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
    
    if total_files == 0:
        print("No training data found. Please run data_collection.py first.")
        return None
    
    print(f"Found {total_files} training samples.")
    
    # Process data files
    processed = 0
    for gesture_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, gesture_class)
        if os.path.isdir(class_dir):
            class_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            print(f"Loading {len(class_files)} samples for gesture '{gesture_class}'")
            
            for sample_file in class_files:
                sample_path = os.path.join(class_dir, sample_file)
                try:
                    sample_data = np.load(sample_path)
                    
                    # Check if data has consistent shape
                    if len(sample_data) > 0:
                        X.append(sample_data)
                        y.append(gesture_class)
                        processed += 1
                except Exception as e:
                    print(f"Error loading {sample_path}: {e}")
    
    print(f"Successfully loaded {processed} out of {total_files} samples.")
    
    if len(X) == 0:
        print("No valid data samples found. Please recollect data.")
        return None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Create label encoder and encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Create and train the classifier
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Feature importance
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.title('Feature Importances')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        print("Feature importance plot saved as 'feature_importances.png'")
    
    # Save the model and label encoder
    with open('gesture_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print("Model saved as 'gesture_model.pkl'")
    print("Label encoder saved as 'label_encoder.pkl'")
    
    return clf, label_encoder

if __name__ == "__main__":
    train_gesture_classifier()