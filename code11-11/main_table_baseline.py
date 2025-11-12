
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings

# Suppress warnings from LightGBM about categorical features
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

def run_baseline():
    """
    Runs a baseline LightGBM model on the main table (application_train.csv)
    to establish a benchmark AUC score.
    """
    print("[Baseline] Starting baseline model evaluation...")

    # 1. Load Data
    try:
        data_path = Path(__file__).parent / "dateset_test" / "application_train.csv"
        df = pd.read_csv(data_path)
        print(f"[Baseline] Successfully loaded data from: {data_path}")
    except FileNotFoundError:
        print(f"[Baseline] Error: Data file not found at {data_path}")
        return

    # 2. Preprocess Target
    if "TARGET" not in df.columns:
        print("[Baseline] Error: Target column 'TARGET' not found in the dataframe.")
        return
        
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    # 3. Preprocess Features
    categorical_features = X.select_dtypes(include=['object']).columns
    
    for col in categorical_features:
        # Use LabelEncoder for categorical features
        le = LabelEncoder()
        # Fit on the entire column to see all possible categories
        # Handle NaNs by temporarily filling them
        # Using a placeholder that's unlikely to exist
        placeholder = '---missing---' 
        X[col] = X[col].astype(str).fillna(placeholder)
        le.fit(X[col])
        X[col] = le.transform(X[col])
        # Optional: If you want to keep NaNs as a separate category, you can adjust
        # For now, LabelEncoder will treat our placeholder as a new category

    print("[Baseline] Data preprocessed.")
    print(f"[Baseline] X shape: {X.shape}")
    print(f"[Baseline] y shape: {y.shape}")

    # 4. Train and Evaluate using Cross-Validation
    # Using 5 splits for a more robust baseline
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    print("\n[Baseline] Starting 5-fold cross-validation...")
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(random_state=42, objective='binary', verbose=-1)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(10, verbose=False)])

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")

    mean_auc = np.mean(aucs)
    print(f"\n[Baseline] Mean AUC: {mean_auc:.4f}")
    return mean_auc

if __name__ == "__main__":
    run_baseline()
