# =========================================================
# train_churn_model.py
# Universal XGBoost training with imbalance handling,
# 3-way split, early stopping, diagnostics & leakage check
# =========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb
from sklearn.utils import resample
import joblib
import re

# ---------------------------------------------------------
# 1️⃣ Load validated data
# ---------------------------------------------------------
base_dir = os.path.expanduser("~/2")
data_path = os.path.join(base_dir, "data", "processed", "validated_data.csv")
model_dir = os.path.join(base_dir, "models", "model_registry")
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(data_path)
print(f"✅ Loaded validated dataset: {data_path}")
print(f"Shape before training: {df.shape}")

# ---------------------------------------------------------
# 2️⃣ Define target + features
# ---------------------------------------------------------
target_col = "account_churn_flag"
if target_col not in df.columns:
    raise ValueError(f"❌ Target column '{target_col}' not found in dataset.")

X = df.drop(columns=[target_col])
y = df[target_col]

# ---------------------------------------------------------
# 3️⃣ Class Imbalance Check + Optional Downsampling
# ---------------------------------------------------------
class_counts = y.value_counts(normalize=True)
print("\n📊 Class Distribution in Target (account_churn_flag):")
print(class_counts)

minority_ratio = class_counts.min()
if minority_ratio < 0.05:
    print("⚠️ Extreme class imbalance detected (<5%). Applying downsampling...")
    df[target_col] = y
    majority = df[df[target_col] == 0]
    minority = df[df[target_col] == 1]
    downsample_factor = 4
    majority_down = resample(
        majority,
        replace=False,
        n_samples=len(minority) * downsample_factor,
        random_state=42
    )
    df = pd.concat([majority_down, minority])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"✅ Downsampled majority class to {len(majority_down)} samples.")
else:
    print("✅ Moderate imbalance handled with class weighting only.")

# ---------------------------------------------------------
# 4️⃣ Use All Features (No Feature Selection)
# ---------------------------------------------------------
print("\n⚙️ Using all validated features (no feature selection).")
final_selected_cols = X.columns.tolist()
print(f"✅ Using all {len(final_selected_cols)} features for training.")

# ---------------------------------------------------------
# 5️⃣ Train / Validation / Test Split (3-way)
# ---------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)
print("\n✅ Data Split Summary:")
print(f"Training set:   {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Testing set:    {X_test.shape}")

# ---------------------------------------------------------
# 6️⃣ Convert to DMatrix for XGBoost
# ---------------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# ---------------------------------------------------------
# 7️⃣ Define Parameters (Imbalance handling)
# ---------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
    "verbosity": 1
}

# ---------------------------------------------------------
# 8️⃣ Train with Early Stopping
# ---------------------------------------------------------
evals_result = {}
eval_list = [(dtrain, "train"), (dval, "validation")]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=400,
    evals=eval_list,
    early_stopping_rounds=30,
    evals_result=evals_result,
    verbose_eval=False
)
print(f"✅ Early Stopping applied. Best iteration: {model.best_iteration}")

# ---------------------------------------------------------
# 9️⃣ Evaluate Performance
# ---------------------------------------------------------
def evaluate_model(model, dtrain, dval, dtest, y_train, y_val, y_test):
    preds = {
        "train": model.predict(dtrain),
        "val": model.predict(dval),
        "test": model.predict(dtest)
    }
    results = {}
    for key, prob in preds.items():
        pred = (prob > 0.5).astype(int)
        auc = roc_auc_score(eval(f"y_{key}"), prob)
        f1 = f1_score(eval(f"y_{key}"), pred)
        results[key] = {"AUC": auc, "F1": f1}
    return results

results = evaluate_model(model, dtrain, dval, dtest, y_train, y_val, y_test)
print("\n================ EVALUATION METRICS ================")
for k, v in results.items():
    print(f"{k.capitalize()} → AUC: {v['AUC']:.4f}, F1: {v['F1']:.4f}")
print("\nDetailed Test Classification Report:")
print(classification_report(y_test, (model.predict(dtest) > 0.5).astype(int)))
print("====================================================")

# ---------------------------------------------------------
# 🔟 Overfitting / Underfitting Diagnostics
# ---------------------------------------------------------
train_auc = results["train"]["AUC"]
val_auc = results["val"]["AUC"]
auc_gap = train_auc - val_auc

if auc_gap > 0.03:
    print("⚠️ Potential OVERFITTING detected — Train AUC much higher than Validation AUC.")
elif train_auc < 0.85 and val_auc < 0.85:
    print("⚠️ UNDERFITTING — both Train and Validation AUC are low.")
else:
    print("✅ Model shows balanced generalization.")

# ---------------------------------------------------------
# 1️⃣1️⃣ Feature Importance + Leakage Detection
# ---------------------------------------------------------
booster = model
importance = booster.get_score(importance_type="gain")
importance_df = pd.DataFrame(
    sorted(importance.items(), key=lambda x: x[1], reverse=True),
    columns=["Feature", "Importance"]
)

print("\n🏆 Top 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

importance_csv = os.path.join(model_dir, "feature_importance.csv")
importance_df.to_csv(importance_csv, index=False)
print(f"📄 Full importance list saved at: {importance_csv}")

leak_keywords = ["churn", "subscription", "reactivation", "cancel", "end", "close", "date", "flag"]
leaky_features = [
    f for f in importance_df["Feature"]
    if any(re.search(k, f, re.IGNORECASE) for k in leak_keywords)
]

if leaky_features:
    print("\n⚠️ POSSIBLE LEAKAGE DETECTED in top features:")
    for f in leaky_features:
        print(f"   - {f}")
    print("👉 Review these columns—they might encode post-churn info.")
else:
    print("✅ No obvious leaky features detected in top predictors.")

# ---------------------------------------------------------
# 1️⃣2️⃣ Learning Curve Visualization
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(evals_result["train"]["auc"], label="Train AUC")
plt.plot(evals_result["validation"]["auc"], label="Validation AUC")
plt.title("Learning Curve: Train vs Validation AUC")
plt.xlabel("Boosting Rounds")
plt.ylabel("AUC Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ---------------------------------------------------------
# 1️⃣2️⃣ SHAP Summary Visualization (Feature Impact)
# ---------------------------------------------------------
import shap

print("\n📊 Generating SHAP summary visualization...")

# Convert training data to array
X_train_sample = X_train.sample(n=min(5000, len(X_train)), random_state=42)

# Initialize TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_sample)

# Plot summary (global importance)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_sample, plot_type="bar", show=False)
plt.title("SHAP Feature Importance Summary (Global Impact)")
plt.tight_layout()
plt.show()

# Detailed beeswarm for feature impact direction
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_train_sample, show=False)
plt.title("SHAP Feature Impact (Direction & Magnitude)")
plt.tight_layout()
plt.show()

print("✅ SHAP visualizations generated successfully.")

# ---------------------------------------------------------
# 1️⃣3️⃣ Save Final Model
# ---------------------------------------------------------
model_path = os.path.join(model_dir, "xgb_churn_model.json")
model.save_model(model_path)
print(f"✅ Final model saved at: {model_path}")


















