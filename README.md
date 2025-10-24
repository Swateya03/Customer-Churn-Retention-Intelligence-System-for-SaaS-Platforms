# Customer-Churn-Retention-Intelligence-System-for-SaaS-Platforms
Our system integrates multi-table SaaS data across accounts, subscriptions, usage, support, and churn to build a 360° view of the customer lifecycle. By linking engagement, satisfaction, and financial outcomes, it forecasts churn risk, quantifies potential MRR/ARR loss, and simulates retention strategies.

---

## Overview

The **Software-as-a-Service (SaaS)** model thrives on recurring subscriptions instead of one-time licenses.  
That means business success depends less on acquiring new customers and more on **keeping existing ones**.  
However, predicting which customers are likely to churn — and understanding *why* — remains a key challenge.

This project provides an end-to-end framework to:
- **Predict** which accounts are at risk of churn  
- **Explain** the behavioral or experience-based drivers behind churn  
- **Forecast** the financial impact on MRR/ARR  
- **Simulate** the ROI of retention initiatives  

By combining analytics, machine learning, and visualization, this solution delivers a full view of customer health and revenue risk.

---

## Business Goals

### 1️ Predict Customer Churn
Build a machine learning model that predicts which accounts are most likely to cancel or not renew.

**Key Outputs**
- Churn probability for each customer  
- Ranked list of high-risk accounts for retention teams  

**Challenges**
- Strong class imbalance (few churners vs active users)  
- Sparse feature usage and feedback data  

---

### 2️ Identify Churn Drivers
Find what behaviors or experiences correlate most strongly with churn.

**Key Insights**
- Analyze support satisfaction, refund trends, and product usage patterns  
- Use NLP on feedback text to detect reasons like *pricing*, *bugs*, or *missing features*

**Why It Matters**
- Helps leadership target the right pain points before customers leave  

---

### 3️ Forecast Financial Impact
Project MRR and ARR loss over time and simulate the benefits of retention actions.

**Key Insights**
- Forecast future churn-related revenue dips  
- Quantify potential revenue retained from churn reduction scenarios  

**Why It Matters**
- Connects data science results directly to business outcomes  

---

## Data Overview

**Dataset:** *RavenStack SaaS Dataset* (simulated multi-table business data)

| Table | Description | Key Fields |
|:------|:-------------|:-----------|
| **accounts.csv** | Customer details | `account_id`, `industry`, `country`, `plan_tier`, `is_trial` |
| **subscriptions.csv** | Subscription and billing lifecycle | `subscription_id`, `account_id`, `mrr_amount`, `churn_flag`, `auto_renew_flag` |
| **feature_usage.csv** | Product usage tracking | `usage_id`, `subscription_id`, `feature_name`, `usage_count`, `usage_duration`, `error_count` |
| **support_tickets.csv** | Support performance and satisfaction | `ticket_id`, `account_id`, `resolution_time`, `priority`, `satisfaction` |
| **churn_events.csv** | Cancellation details and customer feedback | `churn_event_id`, `account_id`, `churn_date`, `reason_code`, `refund_amount`, `feedback_text` |

**Data Relationships**
- `account_id` connects all tables (primary key)  
- `subscription_id` links usage and billing  
- `ticket_id` and `churn_event_id` track post-support and churn events  

---

## Approach

### Step 1 – Data Engineering
- Merge multiple data sources into one consolidated SQL view (SQLite / Snowflake).  
- Aggregate key metrics such as average usage time, satisfaction, and refund amount.  
- Encode categorical features (`industry`, `plan_tier`, `country`).  
- Handle missing values with median or KNN imputation.

---

### Step 2 – Feature Engineering
Derived features:
- **Engagement Score** = normalized `(usage_count + usage_duration)`  
- **Support Efficiency** = `satisfaction / resolution_time`  
- **Revenue Stability** = `(1 – churn_flag) × auto_renew_flag`  
- Text feature extraction using **TF-IDF / BERTopic** for `feedback_text`.

---

### Step 3 – Modeling

| Task | Algorithm | Key Metric |
|------|------------|------------|
| Churn Prediction | XGBoost / CatBoost | ROC-AUC, F1-score |
| Explainability | SHAP / Partial Dependence | Feature Importance (global & local) |
| Revenue Forecasting | Prophet / LSTM | MAPE for MRR/ARR |

---

### Step 4 – Visualization & Insights
Build a **Retention Intelligence Dashboard** (Plotly / Tableau) showing:
- Churn risk by industry, plan, or region  
- Top churn drivers (low usage, poor satisfaction, refunds)  
- MRR & ARR forecast with simulated churn-loss impact  

---

## Results

| Metric | Value | Interpretation |
|:--------|:-------|:---------------|
| **AUC-ROC** | ~0.89 | High accuracy in identifying churners |
| **Recall@10** | ~0.82 | Strong recall of top high-risk customers |
| **Top Drivers** | Low satisfaction, high refunds, low usage | Key churn indicators |
| **Feedback Themes** | “Pricing”, “Support delay”, “Feature gaps” | Most common churn causes |
| **Forecast Error (MAPE)** | ±5.8% | Reliable MRR/ARR forecasting |
| **Retention ROI (Simulated)** | 15% churn reduction → +$200K ARR retained | Tangible financial value |

---

## Insights & Impact

- Transforms raw SaaS data into actionable retention metrics  
- Connects customer experience (support, usage) with revenue impact  
- Enables proactive retention strategies using explainable ML  
- Provides a scalable blueprint for data-driven customer success teams  

---

## Tech Stack

**Languages:** Python, SQL  
**Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Prophet, TensorFlow, BERTopic, SHAP, Plotly, Tableau  
**Database:** SQLite / Snowflake  
**Visualization:** Plotly Dash / Tableau  

---

## Key Takeaways

- Predictive analytics + causal diagnostics = measurable retention value  
- Explainability bridges the gap between data science and business teams  
- Integrated forecasting ties model outputs directly to financial KPIs  
- Real-world frameworks like this drive sustainable SaaS growth  

---

## Author
**Swateya Gupta**  
Delivery Data Scientist | Turing  
M.S. (Research) – IIT Guwahati  

---
