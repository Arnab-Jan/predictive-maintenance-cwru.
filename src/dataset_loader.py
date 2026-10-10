import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.feature_extraction import extract_time_features, extract_freq_features

def process_cwru_mat_file(file_path: str, label_id: int, file_id: str, 
                         window_size: int = 2048, overlap: int = 1024, 
                         fs: int = 48000, windows_per_block: int = 10) -> pd.DataFrame:
    """
    Parses CWRU .mat file, applies overlapping sliding windowing, assigns block_id, 
    and constructs feature DataFrame.
    """
    mat = scipy.io.loadmat(file_path)
    # Search dynamically for Drive End time-series array key
    de_key = [k for k in mat.keys() if 'DE_time' in k][0]
    raw_signal = mat[de_key].flatten()
    
    step = window_size - overlap
    rows = []
    window_count = 0
    
    for start in range(0, len(raw_signal) - window_size + 1, step):
        window = raw_signal[start : start + window_size]
        
        t_feats = extract_time_features(window)
        f_feats = extract_freq_features(window, fs=fs)
        
        # Assign a block ID to group contiguous windows together
        block_number = window_count // windows_per_block
        block_id = f"file_{file_id}_block_{block_number}"
        
        combined = {**t_feats, **f_feats, 'label': label_id, 'block_id': block_id}
        rows.append(combined)
        window_count += 1
        
    return pd.DataFrame(rows)

def build_dataset_and_split(file_map: dict):
    """
    Loads files, concatenates feature dataframes, executes GroupShuffleSplit, 
    and simulates realistic 10:1 class imbalance.
    """
    dfs = []
    for cond_name, info in file_map.items():
        df_cond = process_cwru_mat_file(
            file_path=info['path'], 
            label_id=info['label'], 
            file_id=info['file_id']
        )
        dfs.append(df_cond)
        
    df_all = pd.concat(dfs, ignore_index=True)
    
    X = df_all.drop(columns=['label', 'block_id'])
    y = df_all['label']
    groups = df_all['block_id']
    
    # Leak-free splitting via GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    
    # Simulate factory class imbalance: Drop 90% of fault training samples
    np.random.seed(42)
    normal_mask = (y_train == 0)
    fault_mask = (y_train != 0)
    
    fault_indices = np.where(fault_mask)[0]
    keep_fault_indices = fault_indices[np.random.rand(len(fault_indices)) < 0.10]
    normal_indices = np.where(normal_mask)[0]
    
    imb_train_indices = np.concatenate([normal_indices, keep_fault_indices])
    
    X_train_imb = X_train.iloc[imb_train_indices]
    y_train_imb = y_train.iloc[imb_train_indices]
    
    return X_train_imb, y_train_imb, X_test, y_test
