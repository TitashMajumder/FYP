import sqlite3
from pathlib import Path
from config import SURVEY_SCHEMA


def initialize_database(db_path):
    """
    Creates the database directory, database file,
    and survey table if they don't exist.
    """

    db_path = Path(db_path)

    # Create parent directory if missing
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    try:
        cursor = conn.cursor()
        cursor.execute(SURVEY_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def save_analysis_to_db(db_path, analysis_data):

    db_path = Path(db_path)

    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = None

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        query = """
        INSERT INTO survey (
            timestamp,
            tree_name,
            health,
            confidence,
            reliability,
            latitude,
            longitude,
            details,
            image_files,
            segment_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        data_tuple = (
            analysis_data["timestamp"],
            analysis_data["tree_name"],
            analysis_data["health"],
            analysis_data["confidence"],
            analysis_data["reliability"],
            analysis_data["latitude"],
            analysis_data["longitude"],
            analysis_data["details"],
            analysis_data["image_files"],
            analysis_data.get("segment_path")
        )

        cursor.execute(query, data_tuple)
        conn.commit()

        return True, "Successfully saved to database."

    except Exception as e:

        print(f"Error saving to database: {e}")

        return False, f"Error: {e}"

    finally:

        if conn:
            conn.close()