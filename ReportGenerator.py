import sqlite3
import os

def initialize_database(db_path):
     """
     Creates the database and the 'survey' table if they don't exist.
     This function is safe to call every time the app starts.
     """
     conn = sqlite3.connect(db_path)
     cursor = conn.cursor()
     
     # Define the table structure
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

def save_analysis_to_db(db_path, analysis_data):
     """
     Saves a single analysis result to the SQLite database.

     Args:
          db_path (str): The path to the SQLite database file (e.g., "survey.db").
          analysis_data (dict): A dictionary containing the data to save.
     """
     try:
          conn = sqlite3.connect(db_path)
          cursor = conn.cursor()
          
          # The SQL query to insert data
          query = """
          INSERT INTO survey (
               timestamp, tree_name, health, confidence, reliability,
               latitude, longitude, details, image_files
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
               analysis_data["image_files"]
          )
          
          cursor.execute(query, data_tuple)
          conn.commit()
          conn.close()
          
          return True, "Successfully saved to database."
          
     except Exception as e:
          print(f"Error saving to database: {e}")
          return False, f"Error: {e}"