
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from pathlib import Path

# 1. Load Data
CSV_PATH = Path(__file__).parent / "bank-additional-full.csv"
df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")

# 2. Preprocess Target
if df["y"].dtype == "object":
    df["y"] = (df["y"].astype(str).str.lower() == "yes").astype(int)

X = df.drop("y", axis=1)
y = df["y"]

# 3. Identify Feature Types
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# 4. Preprocess Features
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

print("[Baseline] Data preprocessed.")
print(f"[Baseline] X shape: {X.shape}")
print(f"[Baseline] Categorical features: {list(categorical_features)}")
print(f"[Baseline] Numerical features: {list(numerical_features)}")


# 5. Train and Evaluate
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []

print("\n[Baseline] Starting 5-fold cross-validation...")
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(10, verbose=False)],
              categorical_feature=[X.columns.get_loc(c) for c in categorical_features])

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    aucs.append(auc)
    print(f"  Fold {fold+1} AUC: {auc:.4f}")

print(f"\n[Baseline] Mean AUC: {np.mean(aucs):.4f}")
