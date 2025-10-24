# =========================================================
# validate_data.py
# Comprehensive post-feature-engineering validation:
# - Validates numeric & encoded categorical columns
# - Fixes invalid values (NaN, Inf, negatives)
# - Detects & drops leaky features (refund_amount_usd, subscription_is_trial, escalation_flag)
# - Logs summary with before-after comparison
# =========================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------
# 1️⃣ Load dataset
# ---------------------------------------------------------
base_dir = os.path.expanduser("~/2")
data_path = os.path.join(base_dir, "data", "processed", "final_features.csv")
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(log_dir, f"data_validation_{timestamp}.log")

start_time = datetime.now()
df = pd.read_csv(data_path)
print(f"✅ Loaded data for validation: {data_path}")
print(f"Shape: {df.shape}")

# ---------------------------------------------------------
# 2️⃣ Logging helper
# ---------------------------------------------------------
def log(message):
    print(message)
    with open(log_path, "a") as f:
        f.write(message + "\n")

log(f"🔍 Validation Log Started: {datetime.now()}")
log(f"Dataset Shape: {df.shape}\n")

# ---------------------------------------------------------
# 3️⃣ Pre-validation statistics snapshot
# ---------------------------------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
pre_stats = pd.DataFrame({
    "mean_before": df[numeric_cols].mean(),
    "var_before": df[numeric_cols].var(),
    "missing_before": df[numeric_cols].isna().sum()
})

# ---------------------------------------------------------
# 4️⃣ Validation Function
# ---------------------------------------------------------
def validate_data(df):
    correction_count = 0
    corrections = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    encoded_cols = [col for col in numeric_cols if df[col].nunique() <= 10]
    continuous_cols = [col for col in numeric_cols if col not in encoded_cols]

    log(f"🔢 Numeric columns detected: {len(numeric_cols)}")
    log(f"📊 Encoded categorical columns: {len(encoded_cols)}")
    log(f"📈 Continuous numeric columns: {len(continuous_cols)}\n")

    # Handle infinities and NaNs
    nan_inf_before = df.isna().sum().sum()
    if nan_inf_before > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        log(f"⚠️ Fixed {nan_inf_before} missing/inf values → replaced with 0.")
        correction_count += nan_inf_before

    # Continuous Feature Sanity Checks
    for col in continuous_cols:
        neg_mask = df[col] < 0
        if neg_mask.any():
            msg = f"⚠️ Found {neg_mask.sum()} negative values in {col} → clipped to 0."
            corrections.append(msg)
            df.loc[neg_mask, col] = 0
            correction_count += neg_mask.sum()

    # Encoded Column Integrity Checks
    binary_like = [col for col in encoded_cols if set(df[col].unique()).issubset({0, 1})]
    non_binary = [col for col in encoded_cols if not set(df[col].unique()).issubset({0, 1})]

    if non_binary:
        log(f"⚠️ Non-binary encoded columns found: {non_binary[:10]} ... (showing first 10)")
        for col in non_binary:
            min_val, max_val = df[col].min(), df[col].max()
            if min_val < 0:
                df[col] = df[col].clip(lower=0)
                corrections.append(f"⚙️ {col} had negatives → clipped to 0.")
            if max_val > 10:
                df[col] = df[col].clip(upper=10)
                corrections.append(f"⚙️ {col} exceeded expected range → capped at 10.")

    if corrections:
        for msg in corrections:
            log(msg)
        log("✅ Auto-corrections applied successfully.\n")
    else:
        log("✅ All data integrity checks passed cleanly.\n")

    total_missing = df.isna().sum().sum()
    constant_cols = (df.nunique() == 1).sum()
    log(f"🔍 Missing values remaining: {total_missing}")
    log(f"📏 Constant-value columns: {constant_cols}")
    log(f"⚙️ Binary-like encoded columns: {len(binary_like)}\n")

    return df, correction_count, numeric_cols


df, correction_count, numeric_cols = validate_data(df)

# ---------------------------------------------------------
# 5️⃣ Temporal Leakage Check (refunds, trial flags)
# ---------------------------------------------------------
if "account_churn_flag" in df.columns:
    log("🧭 Running temporal leakage audit...")

    # Convert date-like columns
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except:
            pass
    if date_cols:
        log(f"📅 Converted {len(date_cols)} potential date columns to datetime.")

    # --- Refund leakage ---
    if "refund_amount_usd" in df.columns:
        log("🚨 Dropping refund_amount_usd to eliminate residual financial leakage.")
        df.drop(columns=["refund_amount_usd"], inplace=True)
        if "refund_amount_usd" not in df.columns:
            log("✅ refund_amount_usd successfully removed from dataframe.")

    # --- Subscription trial leakage ---
    if "subscription_is_trial" in df.columns:
        if "start_date" in df.columns and "churn_date" in df.columns:
            try:
                start_dt = pd.to_datetime(df["start_date"], errors="coerce")
                churn_dt = pd.to_datetime(df["churn_date"], errors="coerce")
                trial_after = (df["subscription_is_trial"] == 1) & (churn_dt > start_dt)
                ratio = trial_after.mean()
                log(f"⚠️ subscription_is_trial overlaps post-churn timeline in {ratio:.2%} of records.")
                if ratio > 0.1:
                    log("🚨 Dropping subscription_is_trial (likely post-churn info).")
                    df.drop(columns=["subscription_is_trial"], inplace=True)
                else:
                    log("✅ subscription_is_trial appears temporally consistent.")
            except Exception as e:
                log(f"⚠️ Could not evaluate subscription_is_trial leakage check: {e}")
else:
    log("ℹ️ Target (account_churn_flag) not found — skipping leakage checks.\n")

# ---------------------------------------------------------
# 5️⃣.3 Force-drop known high-risk columns (safety override)
# ---------------------------------------------------------
force_drop = ["subscription_is_trial", "escalation_flag"]
for col in force_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        log(f"🚨 Force-dropped high-risk column: {col}")

# ---------------------------------------------------------
# 6️⃣ Post-validation statistics snapshot (fixed)
# ---------------------------------------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

post_stats = pd.DataFrame({
    "mean_after": df[numeric_cols].mean(),
    "var_after": df[numeric_cols].var(),
    "missing_after": df[numeric_cols].isna().sum()
})

comparison = pre_stats.join(post_stats, how="inner")
comparison["mean_change(%)"] = ((comparison["mean_after"] - comparison["mean_before"]) /
                                (comparison["mean_before"].replace(0, np.nan))) * 100
comparison["var_change(%)"] = ((comparison["var_after"] - comparison["var_before"]) /
                               (comparison["var_before"].replace(0, np.nan))) * 100
comparison.fillna(0, inplace=True)

compare_path = os.path.join(log_dir, f"validation_comparison_{timestamp}.csv")
comparison.to_csv(compare_path)
log(f"📊 Validation comparison saved at: {compare_path}\n")

# ---------------------------------------------------------
# 7️⃣ Optional Binning + Ordinal Encoding
# ---------------------------------------------------------
if "usage_count" in df.columns:
    try:
        df["usage_level"] = pd.qcut(df["usage_count"], q=4,
                                    labels=["Low", "Medium", "High", "Very High"],
                                    duplicates="drop")
    except ValueError:
        log("⚠️ Too many duplicate values in usage_count — switching to fixed-width binning.")
        df["usage_level"] = pd.cut(df["usage_count"], bins=4,
                                   labels=["Low", "Medium", "High", "Very High"])

if "first_response_time_minutes" in df.columns:
    df["response_speed"] = pd.cut(df["first_response_time_minutes"],
                                  bins=[-np.inf, 30, 120, 240, np.inf],
                                  labels=["Fast (<30m)", "Moderate (30–120m)",
                                          "Slow (2–4h)", "Very Slow (>4h)"])

if "account_seats" in df.columns:
    df["account_size"] = pd.cut(df["account_seats"], bins=[-np.inf, 5, 20, 50, np.inf],
                                labels=["Small", "Medium", "Large", "Enterprise"])

ordinal_maps = {
    "usage_level": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
    "response_speed": {"Fast (<30m)": 0, "Moderate (30–120m)": 1,
                       "Slow (2–4h)": 2, "Very Slow (>4h)": 3},
    "account_size": {"Small": 0, "Medium": 1, "Large": 2, "Enterprise": 3},
}

for col, mapping in ordinal_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
        log(f"✅ Ordinally encoded: {col}")

log("✅ Binning transformations added where applicable.")

# ---------------------------------------------------------
# 8️⃣ Save validated dataset
# ---------------------------------------------------------
save_path = os.path.join(base_dir, "data", "processed", "validated_data.csv")
df.to_csv(save_path, index=False)
log(f"✅ Validated dataset saved at: {save_path}")

# ---------------------------------------------------------
# 9️⃣ Summary Report
# ---------------------------------------------------------
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

summary = f"""
================= VALIDATION SUMMARY =================
🕓 Start Time: {start_time}
🕓 End Time:   {end_time}
📦 Total Records: {df.shape[0]}
📊 Total Columns: {df.shape[1]}
🔢 Numeric Columns Checked: {len(numeric_cols)}
🛠️ Total Corrections Applied: {correction_count}
📈 Comparison Report: {compare_path}
⏱️ Total Duration: {duration:.2f} seconds
=======================================================
"""
log(summary)
print(summary)
print(f"📄 Detailed log saved at: {log_path}")











