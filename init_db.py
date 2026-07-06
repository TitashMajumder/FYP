import sqlite3
from pathlib import Path

from config import (
    DB_REPORT_FILE,
    DB_TRAINING_FILE,
    SURVEY_SCHEMA,
    TRAINING_SCHEMA,
)

REPORT_DB_FILE = DB_REPORT_FILE
DATASET_DB_FILE = DB_TRAINING_FILE


def create_databases():

    initialize_report_db()
    init_training_db()


def initialize_report_db():

    db_path = Path(REPORT_DB_FILE)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checking Report Database at: {db_path.resolve()}")

    conn = sqlite3.connect(str(db_path))

    try:
        cursor = conn.cursor()
        cursor.execute(SURVEY_SCHEMA)
        conn.commit()

    finally:
        conn.close()

    print(f"Database '{db_path}' initialized successfully.")


def init_training_db():

    db_path = Path(DATASET_DB_FILE)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checking Training Dataset Database at: {db_path.resolve()}")

    conn = sqlite3.connect(str(db_path))

    try:
        cursor = conn.cursor()
        cursor.execute(TRAINING_SCHEMA)
        conn.commit()

    finally:
        conn.close()

    print(f"Database '{db_path}' initialized successfully.")


if __name__ == "__main__":
    create_databases()