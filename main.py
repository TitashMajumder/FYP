# File: main.py
import os
from AutoCoordinate import get_lat_lon
from AIModel import analyze_tree_health
from MapVisualizer import plot_tree_health_on_gmap
from ReportGenerator import cluster_tree_health, print_summary
from PDFReport import generate_pdf_report

def process_folder(folder_path):
     tree_data = []
     for file in os.listdir(folder_path):
          if file.lower().endswith((".jpg", ".jpeg", ".png")):
               img_path = os.path.join(folder_path, file)
               lat, lon = get_lat_lon(img_path)
               health, conf = analyze_tree_health(img_path)
               tree_data.append({
                    "image": file,
                    "lat": lat,
                    "lon": lon,
                    "health": health,
                    "confidence": conf
               })
     return tree_data

if __name__ == "__main__":
     folder = "tree_images"  # folder with images
     GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"

     data = process_folder(folder)
     print("Processed:", len(data), "images")

     # Plot dynamic Google Map
     plot_tree_health_on_gmap(data, api_key=GOOGLE_MAPS_API_KEY)

     # Cluster and summarize
     df, summary = cluster_tree_health(data)
     print_summary(summary)

     # Generate PDF report
     generate_pdf_report(summary)
