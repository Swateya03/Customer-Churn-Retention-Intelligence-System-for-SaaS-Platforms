# =========================================================
# feature_engineering.py
# Scales numeric features, handles rare categorical values,
# encodes categorical variables, removes leaky features,
# and visualizes scaling impact
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import joblib

# ---------------------------------------------------------
# 1️⃣ Load cleaned dataset
# ---------------------------------------------------------
base_dir = os.path.expanduser("~/2")
data_path = os.path.join(base_dir, "data", "processed", "clean_data.csv")
df = pd.read_csv(data_path)

print(f"✅ Loaded cleaned data: {data_path}")
print(f"Shape before feature scaling: {df.shape}")

# ---------------------------------------------------------
# 2️⃣ Separate numeric and categorical columns
# ---------------------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Drop binary or constant columns (0/1 or single unique value)
binary_like_cols = [col for col in num_cols if df[col].nunique() <= 2]
num_cols = [col for col in num_cols if col not in binary_like_cols]

print(f"Numeric columns considered for scaling: {len(num_cols)}")
print(f"Skipped binary-like columns: {binary_like_cols}")
print(f"Categorical columns detected: {cat_cols}")

# ---------------------------------------------------------
# 3️⃣ Handle rare categories in categorical features (<1%)
# ---------------------------------------------------------
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    rare_labels = freq[freq < 0.01].index
    if len(rare_labels) > 0:
        print(f"⚠️ {col}: {len(rare_labels)} rare categories grouped as 'Other'")
        df[col] = df[col].replace(rare_labels, "Other")

# ---------------------------------------------------------
# 4️⃣ Numeric Feature Scaling (Dynamic based on distribution)
# ---------------------------------------------------------
minmax_scaler = MinMaxScaler()
zscore_scaler = StandardScaler()

scaling_map = {}
scaled_df = df.copy()

for col in num_cols:
    skewness = df[col].skew()
    if abs(skewness) > 2:
        # Heavy-tailed → log transform
        scaled_df[col] = np.log1p(df[col].clip(lower=0))
        scaling_map[col] = "Log Scaling (heavy-tailed)"
    elif 0.5 < abs(skewness) <= 2:
        # Moderately skewed → MinMax
        scaled_df[col] = minmax_scaler.fit_transform(df[[col]])
        scaling_map[col] = "Min-Max Scaling (moderate skew)"
    else:
        # Roughly normal → Z-score
        scaled_df[col] = zscore_scaler.fit_transform(df[[col]])
        scaling_map[col] = "Z-Score Scaling (normal distribution)"

# ---------------------------------------------------------
# 5️⃣ Ordinal + One-Hot Encoding for Categorical Columns
# ---------------------------------------------------------
ordinal_cols = [col for col in ["account_size", "response_speed", "usage_level"] if col in cat_cols]
ordinal_categories = [
    ["Small", "Medium", "Large", "Enterprise"],
    ["Fast (<30m)", "Moderate (30–120m)", "Slow (2–4h)", "Very Slow (>4h)"],
    ["Low", "Medium", "High", "Very High"],
]
ordinal_categories = [cats for cats, col in zip(ordinal_categories, ordinal_cols)]

nominal_cols = [col for col in cat_cols if col not in ordinal_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols),
    ],
    remainder="drop"
)

if len(cat_cols) > 0:
    cat_encoded = preprocessor.fit_transform(df[cat_cols])
    ohe_feature_names = []
    if len(nominal_cols) > 0:
        ohe_feature_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(nominal_cols)

    final_cat_cols = ordinal_cols + list(ohe_feature_names)
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=final_cat_cols, index=df.index)
    scaled_df = pd.concat([scaled_df.drop(columns=cat_cols), cat_encoded_df], axis=1)

    encoder_path = os.path.join(base_dir, "models", "encoders", "feature_encoder.pkl")
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(preprocessor, encoder_path)
    print(f"✅ Encoders saved successfully at: {encoder_path}")

# ---------------------------------------------------------
# 🚫 Leakage Prevention Filter
# ---------------------------------------------------------
print("\n🧹 Running leakage prevention filter...")

# Keywords that could imply leakage
leak_keywords = [
    "churn", "event_id", "cancel", "date", "end", "close",
    "renew", "reactivation", "upgrade", "downgrade","plan_tier", "reason_code", "account_id", "account_name"
]
protected_cols = ["account_churn_flag"]
leaky_cols = [
    col for col in scaled_df.columns
    if any(k in col.lower() for k in leak_keywords)
    and col not in protected_cols
]
if leaky_cols:
    print(f"⚠️ Detected {len(leaky_cols)} potentially leaky columns:")
    for c in leaky_cols:
        print(f"   - {c}")
    
    # Save list of dropped columns for auditing
    dropped_path = os.path.join(base_dir, "logs", "dropped_leaky_features.csv")
    pd.DataFrame(leaky_cols, columns=["Dropped_Leaky_Features"]).to_csv(dropped_path, index=False)
    print(f"📄 Dropped features list exported to: {dropped_path}")

    scaled_df = scaled_df.drop(columns=leaky_cols)
    print("✅ Leaky columns dropped successfully.")
else:
    print("✅ No potential leakage columns found.")

# ---------------------------------------------------------
# 6️⃣ Save scaled + encoded data
# ---------------------------------------------------------
save_path = os.path.join(base_dir, "data", "processed", "final_features.csv")
scaled_df.to_csv(save_path, index=False)
print(f"✅ Final feature dataset saved at: {save_path}")
print(f"Shape after encoding + scaling: {scaled_df.shape}")

# ---------------------------------------------------------
# 7️⃣ Scaling Summary Report
# ---------------------------------------------------------
report_path = os.path.join(base_dir, "logs", "scaling_report.csv")
pd.DataFrame(list(scaling_map.items()), columns=["Feature", "Applied_Scaling"]).to_csv(report_path, index=False)
print(f"📝 Scaling report exported at: {report_path}")

# ---------------------------------------------------------
# 8️⃣ Visualization Section (numeric only)
# ---------------------------------------------------------
if len(num_cols) > 0:
    original_skew = df[num_cols].skew().reset_index()
    original_skew.columns = ['Feature', 'Skewness_Before']

    scaled_skew = scaled_df[num_cols].skew().reset_index()
    scaled_skew.columns = ['Feature', 'Skewness_After']

    skew_df = pd.merge(original_skew, scaled_skew, on='Feature')
    skew_df = skew_df.sort_values(by='Skewness_Before', ascending=False)

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    sns.boxplot(data=df[num_cols])
    plt.title("Boxplot Before Scaling")
    plt.xticks(rotation=45)

    plt.subplot(1,2,2)
    sns.boxplot(data=scaled_df[num_cols])
    plt.title("Boxplot After Scaling")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,6))
    skew_melted = pd.melt(skew_df, id_vars='Feature', value_vars=['Skewness_Before', 'Skewness_After'],
                        var_name='Stage', value_name='Skewness')
    sns.barplot(data=skew_melted, x='Feature', y='Skewness', hue='Stage', palette='coolwarm')
    plt.title("Change in Skewness Before vs After Scaling")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("✅ Visualization complete. Check boxplots and skewness changes above.")
else:
    print("ℹ️ No numeric columns available for skewness visualization.")




