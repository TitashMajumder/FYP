# File: dashboard.py
import os
import time
import sqlite3
import datetime
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw  # Added ImageDraw
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from MapVisualizer import create_health_map
from FuzzyLogic import get_fuzzy_reliability_label
from AIModel import analyze_tree_health, get_treatment_plan, get_gps_from_stamp
from ReportGenerator import initialize_database, save_analysis_to_db
from init_db import init_training_db

# --- 1. STREAMLIT DASHBOARD SETUP ---
st.set_page_config(page_title="Tree Health Dashboard", layout="wide")
st.title("🌳 Tree Health Monitoring Dashboard")

# --- 2. DEFINE THE DB FILE NAME ---
DB_REPORT_FILE = "tree_survey.db"
DB_TRAINING_FILE = "training_dataset.db"

# --- 3. INITIALIZE THE DATABASE ---
initialize_database(DB_REPORT_FILE)
init_training_db()

# --- HELPER: SAVE TO TRAINING DB ---
def save_to_training_db(image_path, tree_name, health_condition):
     """Saves the path and labels to the training dataset database."""
     try:
          if not image_path: return
          conn = sqlite3.connect(DB_TRAINING_FILE)
          cursor = conn.cursor()
          timestamp = datetime.datetime.now().isoformat()
          cursor.execute("INSERT INTO training_data (timestamp, image_path, label_tree_name, label_health_condition) VALUES (?, ?, ?, ?)", 
                         (timestamp, image_path, tree_name, health_condition))
          conn.commit()
          conn.close()
     except Exception as e:
          print(f"Error saving to training DB: {e}")

# --- HELPER: SAVE SEGMENTED IMAGE ---
def save_segmented_image(original_image_path, box_coords, tree_name, timestamp_str):
     """
     Crops the image based on coords and saves it to 'segments' folder.
     Returns the file path of the saved segment.
     """
     try:
          # Validate coordinates
          if not box_coords or len(box_coords) != 4:
               return None
               
          img = Image.open(original_image_path)
          w, h = img.size
          ymin, xmin, ymax, xmax = box_coords
          
          # Convert 0-1000 scale to pixels
          left = (xmin / 1000) * w
          top = (ymin / 1000) * h
          right = (xmax / 1000) * w
          bottom = (ymax / 1000) * h
          
          # Crop the image
          cropped_img = img.crop((left, top, right, bottom))
          
          # Create directory if it doesn't exist
          save_dir = "segments"
          os.makedirs(save_dir, exist_ok=True)
          
          # Create a safe filename: "Mango_20231209_120000.jpg"
          clean_name = "".join(c for c in tree_name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
          clean_time = timestamp_str.replace(':', '').replace('-', '').replace('.', '')
          filename = f"{clean_name}_{clean_time}.jpg"
          
          file_path = os.path.join(save_dir, filename)
          
          # Save file
          cropped_img.save(file_path)
          return file_path
          
     except Exception as e:
          print(f"Error saving segment: {e}")
          return None
   
# --- HELPER FUNCTION TO DRAW BOXES ---
def draw_diagnosis_box(image_path, box_coords, color="red"):
    """
    Draws a bounding box on the image based on Gemini 0-1000 scale coords.
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB to ensure drawing colors work correctly
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Parse [ymin, xmin, ymax, xmax] from 0-1000 scale
        if box_coords and len(box_coords) == 4:
            ymin, xmin, ymax, xmax = box_coords
            
            # Convert to pixels (Dividing by 1000 is CRITICAL)
            left = (xmin / 1000) * width
            top = (ymin / 1000) * height
            right = (xmax / 1000) * width
            bottom = (ymax / 1000) * height
            
            # Draw Rectangle with specified color
            draw.rectangle([left, top, right, bottom], outline=color, width=8)
            
        return img
    except Exception as e:
        print(f"Drawing error: {e}")
        return Image.open(image_path)

# --- 4. CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["Field Survey", "🗺️ Map Visualizer", "📊 Summary"])

# --- 4.A. TAB 1 (Field Survey) ---
with tab1:
     if 'analysis_done' not in st.session_state:
          st.session_state.analysis_done = False
          st.session_state.analysis_results_list = []
          st.session_state.overall_details = ""
     if 'camera_active' not in st.session_state:
          st.session_state.camera_active = False
     if 'manual_lat' not in st.session_state:
          st.session_state.manual_lat = 0.0
     if 'manual_lon' not in st.session_state:
          st.session_state.manual_lon = 0.0

     def clear_old_results():
          st.session_state.analysis_done = False
          st.session_state.analysis_results_list = []
          st.session_state.overall_details = ""
          st.session_state.manual_lat = 0.0
          st.session_state.manual_lon = 0.0

     camera_image = None
     image_inputs = []
     col1, col2 = st.columns(2)
     with col1:
          st.markdown("### 📸 Take Photo(s)")
          # Fixed width deprecation warning
          if st.button("Take Pictures to Upload", width=200, help="Click to activate or deactivate your camera"):
               st.session_state.camera_active = not st.session_state.camera_active
     with col2:
          st.markdown("### 📁 Or Upload Files Manually")
          uploaded_files = st.file_uploader("Drag and drop files here", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed", on_change=clear_old_results)
     if st.session_state.camera_active:
          st.info("Camera is active. You can take one photo at a time.")
          camera_image = st.camera_input("Take a picture of the tree", label_visibility="collapsed", on_change=clear_old_results)
     if camera_image:
          image_inputs = [camera_image]
          st.session_state.camera_active = False
     elif uploaded_files:
          image_inputs = uploaded_files
     else:
          if not st.session_state.camera_active:
               st.info("👆 Please take a picture or upload one or more tree images to start analysis.")

     # --- IMAGE PROCESSING & ANALYSIS ---
     if image_inputs:
          temp_file_paths = []
          image_captions = []
          images_to_display = []
          primary_temp_path = "" 
          os.makedirs("temp", exist_ok=True)
          for i, uploaded_file in enumerate(image_inputs):
               file_name = f"camera_{int(time.time())}.jpg" if uploaded_file.name == "camera.jpg" else uploaded_file.name
               temp_file_path = os.path.join("temp", file_name)
               with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
               if i == 0:
                    primary_temp_path = temp_file_path
               temp_file_paths.append(temp_file_path)
               image_captions.append(file_name)
               images_to_display.append(Image.open(uploaded_file))
          
          # Show thumbnails of uploaded images (Fixed width)
          st.image(images_to_display, caption=image_captions, width=200)

          # --- 2. GPS LOGIC UPDATED ---
          # --- Step 1: Try OCR ---
          if st.session_state.manual_lat == 0.0:
               with st.spinner("Checking for stamped coordinates (OCR)..."):
                    lat, lon = get_gps_from_stamp(primary_temp_path)
               if lat and lon:
                    st.success(f"✅ Auto-Detected GPS (Stamp): {lat:.6f}, {lon:.6f}")
                    st.session_state.manual_lat = lat
                    st.session_state.manual_lon = lon
                    st.rerun() 
          
          # --- Step 2: Show "Get Live Location" button ---
          st.warning("Use your live location or enter coordinates manually.")
          location_data = get_geolocation()
          if location_data and 'coords' in location_data:
               new_lat = location_data['coords']['latitude']
               new_lon = location_data['coords']['longitude']
               if (st.session_state.manual_lat != new_lat or 
                    st.session_state.manual_lon != new_lon):
                    st.session_state.manual_lat = new_lat
                    st.session_state.manual_lon = new_lon
                    st.rerun()
          elif location_data and 'error' in location_data:
               st.toast(f"Location Error: {location_data['error']['message']}", icon="⚠️")

          # --- Step 3: ALWAYS show the manual entry expander ---
          with st.expander("Enter/Confirm Coordinates", expanded=True):
               manual_lat = st.number_input("Latitude", format="%.6f", value=st.session_state.manual_lat)
               manual_lon = st.number_input("Longitude", format="%.6f", value=st.session_state.manual_lon)
               if manual_lat != st.session_state.manual_lat:
                    st.session_state.manual_lat = manual_lat
               if manual_lon != st.session_state.manual_lon:
                    st.session_state.manual_lon = manual_lon
          if st.session_state.manual_lat != 0.0:
               st.info(f"📍 Using selected coordinates: ({st.session_state.manual_lat:.6f}, {st.session_state.manual_lon:.6f})")
          
          if st.button("🧠 Analyze Tree Health"):
               with st.spinner("Analyzing all images..."):
                    results_list, overall_details = analyze_tree_health(temp_file_paths)
                    st.session_state.analysis_results_list = results_list
                    st.session_state.overall_details = overall_details
                    st.session_state.analysis_done = True
                    if not results_list:
                         st.warning(f"Analysis complete, but no specific plants were identified. Details: {overall_details}")
                    timestamp = datetime.datetime.now().isoformat()
                    saved_count = 0
                    for res in results_list:
                         name = res.get("tree_name", "Unknown")
                         health = res.get("health_condition", "Unknown")
                         confidence = res.get("confidence_percent", 0)
                         reliability = get_fuzzy_reliability_label(confidence)
                         details = res.get("brief_analysis", "No details").replace('\n', ' ')
                         data_to_save = {
                         "timestamp": timestamp,
                         "tree_name": name,
                         "health": health,
                         "confidence": confidence,
                         "reliability": reliability,
                         "latitude": st.session_state.manual_lat,
                         "longitude": st.session_state.manual_lon,
                         "details": details,
                         "image_files": ", ".join(image_captions)
                         }
                         saved, message = save_analysis_to_db(DB_REPORT_FILE, data_to_save)
                         if saved:
                              saved_count += 1
                    if saved_count > 0:
                         st.success(f"✅ Analysis complete! {saved_count} tree(s) identified and saved to report.")
                    else:
                         st.error(f"Failed to save any results to report.")
     
     # --- DISPLAY RESULTS ---
     if st.session_state.analysis_done:
          results_list = st.session_state.analysis_results_list
          overall_details = st.session_state.overall_details
          st.subheader("Overall Analysis")
          st.markdown(overall_details)
          st.markdown("---")
          if not results_list:
               st.info("No individual plants were identified in this analysis.")
          else:
               st.subheader(f"Identified {len(results_list)} Plant(s)")
               
               for i, res in enumerate(results_list):
                    tree_name = res.get('tree_name', 'Unknown')
                    health = res.get('health_condition', 'Unknown')
                    confidence = res.get('confidence_percent', 0)
                    reliability = get_fuzzy_reliability_label(confidence)
                    brief_analysis = res.get('brief_analysis', 'No details')
                    
                    # GET COORDINATES FOR THE BOX
                    box_coords = res.get('diseased_area_box', None)
                    
                    # GET IMAGE INDEX
                    img_idx = res.get('image_index', 0)
                    if img_idx < len(temp_file_paths):
                        img_path = temp_file_paths[img_idx]
                    else:
                        img_path = temp_file_paths[0]
                    
                    # DETERMINE BOX COLOR
                    if health == "Healthy":
                        box_color = "#00FF00" # Green
                    else:
                        box_color = "#FF0000" # Red

                    with st.container(border=True):
                         st.subheader(f"Result {i+1}: {tree_name}")
                         
                         col_img, col_data = st.columns([1, 2])
                         
                         with col_img:
                             # DRAW THE BOX IF COORDS EXIST
                             if box_coords:
                                 # Pass box_color to the function
                                 annotated_img = draw_diagnosis_box(img_path, box_coords, color=box_color)
                                 # Fixed width to 350px for smaller image
                                 st.image(annotated_img, caption=f"Visual Diagnosis: {health}", width=350)
                             else:
                                 st.image(img_path, caption="Original Image", width=350)

                         with col_data:
                             c1, c2, c3 = st.columns(3)
                             c1.metric("🩺 Health", health)
                             c2.metric("🎯 Confidence", f"{confidence}%")
                             c3.metric("📊 Reliability", reliability)
                             
                             st.markdown(f"**Brief Analysis:** {brief_analysis}")
                             
                             if st.button(f"👩‍⚕️ Get Treatment Plan", key=f"treat_{i}"):
                                  with st.spinner(f"Generating plan for {tree_name}..."):
                                       plan = get_treatment_plan(tree_name, health, brief_analysis)
                                       st.subheader(f"Recommended Plan for {tree_name}")
                                       st.markdown(plan)
                                       
# --- 4.B. TAB 2 (Map Visualizer) ---
with tab2:
     st.header("🗺️ Tree Health Map")
     if os.path.exists(DB_REPORT_FILE):
          health_map = create_health_map(DB_REPORT_FILE)
          st.info("Click on a leaf icon to see the analysis details.")
          st_folium(health_map, use_container_width=True, height=600)
     else:
          st.info("No survey data found. Go to the 'Field Survey' tab to save your first analysis.")

# --- 4.C. TAB 3 (Summary) ---
with tab3:
     st.header("📊 Survey Summary")
     if not os.path.exists(DB_REPORT_FILE):
          st.info("No survey data found. Go to the 'Field Survey' tab to save your first analysis.")
     else:
          conn = sqlite3.connect(DB_REPORT_FILE)
          df = pd.read_sql("SELECT * FROM survey", conn)
          conn.close()
          if df.empty:
               st.info("Survey report is empty. Please save some analysis results.")
          else:
               st.subheader("Key Metrics")
               col1_sum, col2_sum, col3_sum = st.columns(3)
               col1_sum.metric("Total Trees Surveyed", len(df))
               try:
                    most_common_issue = df[df['health'] != 'Healthy']['health'].mode()[0]
               except KeyError:
                    most_common_issue = "N/A"
               col2_sum.metric("Most Common Issue", most_common_issue)
               col3_sum.metric("Unique Tree Types", df['tree_name'].nunique())
               st.markdown("---")
               st.subheader("Health Condition Breakdown")
               health_counts = df['health'].value_counts()
               st.bar_chart(health_counts)
               st.subheader("Reliability Breakdown")
               reliability_counts = df['reliability'].value_counts()
               st.bar_chart(reliability_counts)
               st.markdown("---")
               st.subheader("Raw Survey Data")
               st.dataframe(df)
               st.markdown("---")
               st.subheader("⚠️ Danger Zone")
               with st.expander("Clear All Survey Data"):
                    st.warning("This action is permanent and cannot be undone. All survey data will be deleted.")
                    if st.button("I understand, delete all data", type="primary", key="delete_all_data"):
                         try:
                              conn = sqlite3.connect(DB_REPORT_FILE)
                              cursor = conn.cursor()
                              cursor.execute("DELETE FROM survey")
                              cursor.execute("DELETE FROM sqlite_sequence WHERE name='survey'")
                              conn.commit()
                              conn.close()
                              st.success("All survey data has been successfully cleared.")
                              st.rerun() 
                         except Exception as e:
                              st.error(f"Error clearing database: {e}")