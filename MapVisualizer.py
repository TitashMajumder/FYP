import pandas as pd # type: ignore
import folium # type: ignore
import os
import sqlite3
from folium.plugins import MarkerCluster # type: ignore

def create_health_map(db_path):
     """
     Reads survey data from a SQLite DB and creates a Folium map
     with clustered, color-coded, clickable markers.
     """
     if not os.path.exists(db_path):
          return folium.Map(location=[20.5937, 78.9629], zoom_start=5) 
          
     conn = sqlite3.connect(db_path)
     try:
          df = pd.read_sql("SELECT * FROM survey", conn)
     except Exception as e:
          print(f"Error reading database: {e}. Displaying empty map.")
          conn.close()
          return folium.Map(location=[20.5937, 78.9629], zoom_start=5)
          
     conn.close()
     
     df_gps = df[(df['latitude'] != 0.0) & (df['longitude'] != 0.0)]

     if df_gps.empty:
          return folium.Map(location=[20.5937, 78.9629], zoom_start=5)

     center_lat = df_gps['latitude'].mean()
     center_lon = df_gps['longitude'].mean()
     m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

     # --- 2. CREATE A MARKER CLUSTER ---
     marker_cluster = MarkerCluster().add_to(m)

     color_map = {
          "Healthy": "green",
          "Stressed": "orange",
          "Diseased": "red",
          "Critical": "darkred"
     }

     for _, row in df_gps.iterrows():
          health = row['health']
          color = color_map.get(health, "gray")
          
          popup_html = f"""
          <b>Tree Name:</b> {row['tree_name']}<br>
          <b>Health:</b> {row['health']} (Reliability: {row['reliability']})<br>
          <b>Confidence:</b> {row['confidence']}%<br>
          <b>Timestamp:</b> {row['timestamp']}<br>
          <hr>
          <b>Details:</b> {row['details']}
          """
          popup = folium.Popup(popup_html, max_width=300)
          
          # --- 3. ADD THE MARKER TO THE CLUSTER, NOT THE MAP ---
          folium.Marker(
               location=[row['latitude'], row['longitude']],
               popup=popup,
               icon=folium.Icon(color=color, icon='leaf', prefix='fa')
          ).add_to(marker_cluster)  # <--- THIS IS THE CHANGE
     return m