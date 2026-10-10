import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src.dataset_loader import build_dataset_and_split

def main():
    # Define dataset paths and label metadata
    file_map = {
        'normal': {'path': 'data/raw/98.mat', 'label': 0, 'file_id': '98'},
        'inner_race': {'path': 'data/raw/108.mat', 'label': 2, 'file_id': '108'},
        'ball': {'path': 'data/raw/122.mat', 'label': 1, 'file_id': '122'},
        'outer_race': {'path': 'data/raw/133.mat', 'label': 3, 'file_id': '133'}
    }

    print("Extracting features and performing leak-free group split...")
    X_train, y_train, X_test, y_test = build_dataset_and_split(file_map)

    print(f"Dataset split complete: {len(X_train)} training samples, {len(X_test)} testing samples.")
    print("Training set class breakdown:\n", y_train.value_counts())

    # Model 1: Standard Unweighted Baseline
    print("\nTraining Baseline Random Forest...")
    baseline_rf = RandomForestClassifier(random_state=42)
    baseline_rf.fit(X_train, y_train)
    y_pred_base = baseline_rf.predict(X_test)

    print("\n=== BASELINE MODEL REPORT ===")
    print(classification_report(y_test, y_pred_base, target_names=['Normal', 'Ball', 'Inner', 'Outer']))

    # Model 2: Imbalance-Aware Class-Weighted Random Forest
    print("\nTraining Class-Weighted Random Forest...")
    weighted_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    weighted_rf.fit(X_train, y_train)
    y_pred_weighted = weighted_rf.predict(X_test)

    print("\n=== CLASS-WEIGHTED MODEL REPORT ===")
    print(classification_report(y_test, y_pred_weighted, target_names=['Normal', 'Ball', 'Inner', 'Outer']))

    # Export trained model artifact
    joblib.dump(weighted_rf, 'bearing_fault_rf_model.pkl')
    print("\nTrained model exported successfully to 'bearing_fault_rf_model.pkl'.")

if __name__ == "__main__":
    main()
