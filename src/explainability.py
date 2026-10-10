import joblib
import shap
import matplotlib.pyplot as plt
from src.dataset_loader import build_dataset_and_split

def generate_shap_plots():
    file_map = {
        'normal': {'path': 'data/raw/98.mat', 'label': 0, 'file_id': '98'},
        'inner_race': {'path': 'data/raw/108.mat', 'label': 2, 'file_id': '108'},
        'ball': {'path': 'data/raw/122.mat', 'label': 1, 'file_id': '122'},
        'outer_race': {'path': 'data/raw/133.mat', 'label': 3, 'file_id': '133'}
    }

    _, _, X_test, _ = build_dataset_and_split(file_map)
    model = joblib.load('bearing_fault_rf_model.pkl')

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot and save feature importance summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    print("SHAP summary plot saved to 'shap_summary.png'")

if __name__ == '__main__':
    generate_shap_plots()
