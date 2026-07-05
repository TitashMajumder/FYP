import sqlite3
import os
from config import DB_REPORT_FILE, DB_TRAINING_FILE, SURVEY_SCHEMA, TRAINING_SCHEMA

REPORT_DB_FILE  = DB_REPORT_FILE   # kept for backward compat
DATASET_DB_FILE = DB_TRAINING_FILE # kept for backward compat

def create_databases():
    """Creates both SQLite databases and their required tables."""
    
    # --- 1. Initialize Report Database ---
    print(f"Checking Report Database at: {os.path.abspath(REPORT_DB_FILE)}")
    conn_report = sqlite3.connect(REPORT_DB_FILE)
    cursor_report = conn_report.cursor()
    
    # Table for dashboard reports
    cursor_report.execute(SURVEY_SCHEMA)
    conn_report.commit()
    conn_report.close()
    print(f"✅ Database '{REPORT_DB_FILE}' initialized successfully.")

    # --- 2. Initialize Training Dataset Database ---
    init_training_db()

def init_training_db():
    """Initializes the training dataset database."""
    print(f"Checking Training Dataset Database at: {os.path.abspath(DATASET_DB_FILE)}")
    conn_dataset = sqlite3.connect(DATASET_DB_FILE)
    cursor_dataset = conn_dataset.cursor()

    # Table specifically for training data (Image + Label)
    cursor_dataset.execute(TRAINING_SCHEMA)
    conn_dataset.commit()
    conn_dataset.close()
    print(f"✅ Database '{DATASET_DB_FILE}' initialized successfully.")

if __name__ == "__main__":
    create_databases()