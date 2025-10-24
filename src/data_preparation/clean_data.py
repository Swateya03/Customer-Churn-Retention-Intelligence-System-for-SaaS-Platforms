# =========================================================
# clean_data.py
# Cleans merged_customer_data.csv based on predefined rules
# =========================================================

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1️⃣ Load dataset
# ---------------------------------------------------------
base_dir = os.path.expanduser("~/2")
data_path = os.path.join(base_dir, "data", "processed", "merged_customer_data.csv")
df = pd.read_csv(data_path)

print(f"✅ Loaded dataset: {data_path}")
print(f"Shape before cleaning: {df.shape}")

# ---------------------------------------------------------
# 2️⃣ Drop columns with very high missing percentage
# ---------------------------------------------------------
df.drop(columns=['end_date'], inplace=True, errors='ignore')

# ---------------------------------------------------------
# 3️⃣ Mode imputation for categorical or flag columns
# ---------------------------------------------------------
mode_impute_cols = [
    'usage_id', 'usage_date', 'feature_name', 'ticket_id',
    'submitted_at', 'closed_at', 'priority',
    'is_beta_feature', 'escalation_flag',
    'preceding_downgrade_flag', 'preceding_upgrade_flag', 'is_reactivation'
]
for col in mode_impute_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# ---------------------------------------------------------
# 4️⃣ Add "Unknown" or empty string for text/category columns
# ---------------------------------------------------------
unknown_fill_cols = ['churn_event_id', 'churn_date', 'reason_code']
for col in unknown_fill_cols:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)

if 'feedback_text' in df.columns:
    df['feedback_text'].fillna("", inplace=True)

# ---------------------------------------------------------
# 5️⃣ Numeric mean/median imputation
# ---------------------------------------------------------
median_impute_cols = [
    'usage_count', 'usage_duration_secs', 'error_count',
    'resolution_time_hours', 'first_response_time_minutes',
    'refund_amount_usd'
]
for col in median_impute_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# ---------------------------------------------------------
# 6️⃣ Advanced imputation for satisfaction_score using KNN
# ---------------------------------------------------------
if 'satisfaction_score' in df.columns:
    knn = KNNImputer(n_neighbors=5)
    df[['satisfaction_score']] = knn.fit_transform(df[['satisfaction_score']])

# ---------------------------------------------------------
# 7️⃣ OUTLIER HANDLING
# ---------------------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Handle based on column meaning
    if col in ['usage_count', 'usage_duration_secs', 'error_count',
               'resolution_time_hours', 'first_response_time_minutes']:
        df[col] = df[col].clip(lower, upper)
    elif col == 'refund_amount_usd':
        df[col] = np.log1p(df[col])  # log transform to reduce skew
    elif col == 'satisfaction_score':
        df[col] = df[col].clip(1, 5)  # bound scale

print("✅ Outlier handling completed.")

# ---------------------------------------------------------
# 8️⃣ LOG TRANSFORMATION FOR HEAVY-TAILED FEATURES
# ---------------------------------------------------------
log_transform_cols = ['mrr_amount', 'arr_amount', 'usage_duration_secs']
for col in log_transform_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])
        print(f"🔹 Log-transformed column: {col}")

print("✅ Log transformation applied to revenue and duration features.")

# ---------------------------------------------------------
# 9️⃣ Save cleaned dataset
# ---------------------------------------------------------
save_path = os.path.join(base_dir, "data", "processed", "clean_data.csv")
df.to_csv(save_path, index=False)
print(f"✅ Final cleaned data saved at: {save_path}")
print(f"Shape after cleaning: {df.shape}")

# ---------------------------------------------------------
# 🔍 Optional Visualization
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.title("Boxplot After Outlier + Log Handling")
plt.xticks(rotation=45)
plt.show()


