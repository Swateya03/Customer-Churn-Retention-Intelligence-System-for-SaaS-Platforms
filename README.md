# Customer-Churn-Retention-Intelligence-System-for-SaaS-Platforms
Our system integrates multi-table SaaS data across accounts, subscriptions, usage, support, and churn to build a 360° view of the customer lifecycle. By linking engagement, satisfaction, and financial outcomes, it forecasts churn risk, quantifies potential MRR/ARR loss, and simulates retention strategies.

## Introduction

The **Software-as-a-Service (SaaS)** model has transformed how organizations deliver and consume software.  
Unlike one-time licenses, SaaS businesses rely on **recurring subscriptions**, meaning their growth depends heavily on **customer retention** rather than acquisition.  
In this context, **customer churn** — when a user cancels or fails to renew — poses a serious threat to **Monthly Recurring Revenue (MRR)** and **Annual Recurring Revenue (ARR)**.

### Why Churn Prediction Matters
Predicting churn allows SaaS companies to:
- Identify at-risk customers early  
- Personalize retention campaigns  
- Reduce revenue leakage and increase **Customer Lifetime Value (CLV)**  
- Optimize **Net Revenue Retention (NRR)**  

### Prior Work
Most previous research (in telecom or banking) used **single-source transactional datasets**, limiting contextual insight.  
Typical issues include:
- Narrow focus on classification accuracy  
- Ignoring **feature usage** and **support experience**  
- No modeling of **financial impact** such as MRR/ARR forecasting  

### Our Contribution
We propose an **integrated SaaS Churn & Retention Intelligence Framework** that:
1. **Predicts** which customers are likely to churn (Predictive Layer)  
2. **Explains** the behavioral & experiential causes (Diagnostic Layer)  
3. **Simulates** retention interventions and revenue impact (Prescriptive Layer)  

This system combines **multi-table operational data** — `accounts`, `subscriptions`, `feature_usage`, `support_tickets`, and `churn_events` — to deliver a 360° view of customer lifecycle health.

---

## Problem Formulation

### Overall Goal
Design an **end-to-end data-driven framework** that predicts churn, identifies its root causes, and quantifies the financial effect on revenue.

---

### Problem 1 – Predictive Modeling of Customer Churn

**Statement:**  
Develop a machine-learning model that predicts the likelihood of churn using multi-source SaaS data.

**Objectives:**
- Predict `churn_flag` at the account level  
- Rank customers by churn probability  

**Assumptions:**
- Each `account_id` is unique and independent  
- Most recent `subscription` defines churn status  

**Constraints:**
- Severe class imbalance (fewer churners)  
- Synthetic dataset lacks daily granularity  

---

### Problem 2 – Diagnostic Analysis of Churn Drivers

**Statement:**  
Identify behavioral and experiential factors contributing to churn.

**Objectives:**
- Quantify impact of **usage frequency**, **satisfaction**, and **refunds**  
- Perform NLP on `feedback_text` to classify churn reasons  

**Assumptions:**
- Low engagement and satisfaction → higher churn risk  
- Feedback text reliably reflects sentiment  

**Constraints:**
- Limited examples for each churn reason  

---

### Problem 3 – Forecasting and Revenue Impact Simulation

**Statement:**  
Forecast future MRR and ARR and simulate financial loss from churn.

**Objectives:**
- Predict revenue trends using **Prophet** or **LSTM**  
- Compute **NRR** and simulate impact of retention actions  

**Assumptions:**
- Past churn and upgrades represent future patterns  

**Constraints:**
- Must aggregate data monthly for forecasting  

---

## 📊 Dataset Description

The **RavenStack Dataset** simulates a realistic SaaS environment through **five relational tables**.

| Table | Description | Key Fields |
|:------|:-------------|:-----------|
| **accounts.csv** | Customer metadata (industry, region, plan, referral source) | `account_id`, `industry`, `country`, `plan_tier`, `is_trial` |
| **subscriptions.csv** | Subscription lifecycle & billing | `subscription_id`, `account_id`, `mrr_amount`, `arr_amount`, `upgrade_flag`, `churn_flag`, `billing_freq`, `auto_renew_flag` |
| **feature_usage.csv** | Daily product usage | `usage_id`, `subscription_id`, `feature_name`, `usage_count`, `usage_duration`, `error_count`, `is_beta_feature` |
| **support_tickets.csv** | Support activity & satisfaction | `ticket_id`, `account_id`, `resolution_time`, `priority`, `satisfaction`, `escalation_flag` |
| **churn_events.csv** | Churn details & feedback | `churn_event_id`, `account_id`, `churn_date`, `reason_code`, `refund_amount`, `feedback_text` |

**Relational Keys**
- `account_id` → primary key linking all tables  
- `subscription_id` → one-to-many with `account_id`  
- `usage_id` → foreign key to `subscription_id`  
- `ticket_id`, `churn_event_id` → linked to `account_id`  

---

## ⚙️ Methodology

### Step 1 – Data Engineering
1. Load all CSVs into a SQL environment (SQLite / Snowflake).  
2. Join tables using `account_id` and `subscription_id`.  
3. Aggregate metrics: mean `usage_duration`, `error_count`, `satisfaction`, `refund_amount`.  
4. Encode categoricals (`industry`, `country`, `plan_tier`).  
5. Handle missing data with **median** / **KNN imputation**.

---

### Step 2 – Feature Engineering
Derived features:
- **Engagement Score = normalized (usage_count + usage_duration)**  
- **Support Efficiency Index = satisfaction / resolution_time**  
- **Revenue Stability = (1 – churn_flag) × auto_renew_flag**  
- Extract text topics via **TF-IDF / BERTopic**

---

### Step 3 – Modeling

| Task | Algorithm | Evaluation Metrics |
|:-----|:-----------|:------------------|
| Churn Prediction | XGBoost / CatBoost | AUC-ROC , F1-Score |
| Explainability | SHAP , PDP | Feature Importance (global & local) |
| Forecasting | Prophet / LSTM | MAPE for MRR / ARR |
| Text Analytics | TF-IDF + LDA / BERTopic | Topic Coherence & Keyword Weights |

---

### Step 4 – Visualization & Reporting
Create an interactive **Retention Command Center Dashboard** (Tableau / Plotly Dash):
- Churn risk by industry, region, plan tier  
- Feature usage heatmaps and support trends  
- Forecasted MRR and churn-loss projection  
- Word cloud of feedback themes  

---

## 📈 Results (Expected)

| Metric | Value | Interpretation |
|:--------|:-------|:---------------|
| **AUC-ROC** | 0.89 | High predictive accuracy |
| **Recall@10** | 0.82 | Detects most high-risk customers |
| **Top Drivers** | Satisfaction ↓, Usage ↓, Refund ↑, Plan Tier | Behavioral + financial factors |
| **Feedback Themes** | “Pricing”, “Missing Features”, “Support Delay” | Key churn causes |
| **MRR Forecast MAPE** | ± 5.8 % | Reliable revenue forecast |
| **Simulated Impact** | 15 % churn reduction → +$200 K ARR retained | Retention ROI projection |

---

## Conclusion

This project builds an **end-to-end SaaS Retention Intelligence System** combining:
- **Predictive modeling** → *Who will churn?*  
- **Diagnostic analytics** → *Why do they churn?*  
- **Revenue simulation** → *How much revenue can be saved?*  

By integrating **multi-table analytics** and **machine learning**, the system enables proactive, explainable, and financially meaningful retention decisions in SaaS environments.
