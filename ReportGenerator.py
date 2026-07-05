import sqlite3
from config import SURVEY_SCHEMA

def initialize_database(db_path):
     """
     Creates the database and the 'survey' table if they don't exist.
     This function is safe to call every time the app starts.
     """
     conn = sqlite3.connect(db_path)
     cursor = conn.cursor()
     # Define the table structure
     cursor.execute(SURVEY_SCHEMA)
     conn.commit()
     conn.close()

def save_analysis_to_db(db_path, analysis_data):
     """
     Saves a single analysis result to the SQLite database.
     Args:
          db_path (str): The path to the SQLite database file (e.g., "survey.db").
          analysis_data (dict): A dictionary containing the data to save.
     """
     conn = None
     try:
          conn = sqlite3.connect(db_path)
          cursor = conn.cursor()
          # The SQL query to insert data
          query = """
          INSERT INTO survey (
               timestamp, tree_name, health, confidence, reliability,
               latitude, longitude, details, image_files, segment_path
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """
          # Create a tuple of values in the correct order
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