# File: config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DB_REPORT_FILE   = "database/tree_survey.db"
DB_TRAINING_FILE = "database/training_dataset.db"

SURVEY_SCHEMA = """
CREATE TABLE IF NOT EXISTS survey (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    tree_name    TEXT,
    health       TEXT,
    confidence   INTEGER,
    reliability  TEXT,
    latitude     REAL,
    longitude    REAL,
    details      TEXT,
    image_files  TEXT,
    segment_path TEXT
)
"""

TRAINING_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_data (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp              TEXT,
    image_path             TEXT,
    label_tree_name        TEXT,
    label_health_condition TEXT
)
"""