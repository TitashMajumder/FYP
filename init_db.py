import sqlite3
import os

# The filename used in your dashboard.py
DB_FILE = "dataset.db"

def create_database():
    """Creates the SQLite database and the required table."""
    
    # Connects to the database (creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    print(f"Checking database at: {os.path.abspath(DB_FILE)}")

    # Create the 'survey' table based on your schema
    cursor.execute("""
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
        image_files TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database 'dataset.db' initialized successfully.")

if __name__ == "__main__":
    create_database()