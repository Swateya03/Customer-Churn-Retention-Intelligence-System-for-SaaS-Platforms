import os
import pandas as pd
import sqlite3
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    logger.info(f"Connected to SQLite DB at {db_path}")
    return conn

def load_csvs_to_sqlite(raw_dir: str, conn):
    for file in os.listdir(raw_dir):
        if file.endswith(".csv") and not file.endswith(":Zone.Identifier"):
            table_name = file.replace("ravenstack_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(raw_dir, file))
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info(f"Loaded {file} into table '{table_name}' ({len(df)} rows)")
    logger.info("All CSVs successfully loaded into SQLite.")

def merge_tables(conn):
    query = """
    SELECT 
        a.account_id,
        a.account_name,
        a.industry,
        a.country,
        a.signup_date,
        a.referral_source,
        a.plan_tier AS account_plan_tier,
        a.seats AS account_seats,
        a.is_trial AS account_is_trial,
        a.churn_flag AS account_churn_flag,

        s.subscription_id,
        s.start_date,
        s.end_date,
        s.plan_tier AS subscription_plan_tier,
        s.mrr_amount,
        s.arr_amount,
        s.is_trial AS subscription_is_trial,
        s.upgrade_flag,
        s.downgrade_flag,
        s.churn_flag AS subscription_churn_flag,
        s.billing_frequency,
        s.auto_renew_flag,

        u.usage_id,
        u.usage_date,
        u.feature_name,
        u.usage_count,
        u.usage_duration_secs,
        u.error_count,
        u.is_beta_feature,

        t.ticket_id,
        t.submitted_at,
        t.closed_at,
        t.resolution_time_hours,
        t.priority,
        t.first_response_time_minutes,
        t.satisfaction_score,
        t.escalation_flag,

        c.churn_event_id,
        c.churn_date,
        c.reason_code,
        c.refund_amount_usd,
        c.preceding_downgrade_flag,
        c.preceding_upgrade_flag,
        c.is_reactivation,
        c.feedback_text

    FROM accounts a
    LEFT JOIN subscriptions s ON a.account_id = s.account_id
    LEFT JOIN feature_usage u ON s.subscription_id = u.subscription_id
    LEFT JOIN support_tickets t ON a.account_id = t.account_id
    LEFT JOIN churn_events c ON a.account_id = c.account_id;
    """
    df_merged = pd.read_sql_query(query, conn)
    logger.info(f"Merged dataset shape: {df_merged.shape}")
    return df_merged

def main():
    db_config = load_config("config/database.yaml")["sqlite"]
    path_config = load_config("config/paths.yaml")["paths"]

    conn = create_connection(db_config["database_path"])
    load_csvs_to_sqlite(path_config["raw_data_dir"], conn)
    df_merged = merge_tables(conn)

    os.makedirs(path_config["processed_data_dir"], exist_ok=True)
    df_merged.to_csv(path_config["merged_output"], index=False)
    logger.info(f"Merged dataset saved at {path_config['merged_output']}")
    conn.close()

if __name__ == "__main__":
    main()


