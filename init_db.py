import sqlite3
import os

# Database 1: For the User Dashboard (Reports)
REPORT_DB_FILE = "tree_survey.db"

# Database 2: For Future Model Training (Dataset)
DATASET_DB_FILE = "training_dataset.db"

def create_databases():
    """Creates both SQLite databases and their required tables."""
    
    # --- 1. Initialize Report Database ---
    print(f"Checking Report Database at: {os.path.abspath(REPORT_DB_FILE)}")
    conn_report = sqlite3.connect(REPORT_DB_FILE)
    cursor_report = conn_report.cursor()
    
    # Table for dashboard reports
    cursor_report.execute("""
    CREATE TABLE IF NOT EXISTS survey (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        tree_name TEXT,
        health TEXT,
        confidence INTEGER,
        reliability TEXT,
        latitude REAL,
        longitude REAL,
        details TEXT,
        image_files TEXT,
        segment_path TEXT
    )
    """)
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
    cursor_dataset.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        image_path TEXT,
        label_tree_name TEXT,
        label_health_condition TEXT
    )
    """)
    conn_dataset.commit()
    conn_dataset.close()
    print(f"✅ Database '{DATASET_DB_FILE}' initialized successfully.")

if __name__ == "__main__":
    create_databases()