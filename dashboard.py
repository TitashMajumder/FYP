# File: dashboard.py
import os
import time
import sqlite3
import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# --- Import all your custom modules ---
from AutoCoordinate import get_lat_lon
from MapVisualizer import create_health_map
from FuzzyLogic import get_fuzzy_reliability_label
from AIModel import analyze_tree_health, get_treatment_plan, get_gps_from_stamp
from ReportGenerator import initialize_database, save_analysis_to_db

# --- STREAMLIT DASHBOARD SETUP ---
st.set_page_config(page_title="Tree Health Dashboard", layout="wide")
st.title("🌳 Tree Health Monitoring Dashboard")

# --- 1. DEFINE THE DB FILE NAME ---
DB_REPORT_FILE = "tree_survey.db"

# --- 2. INITIALIZE THE DATABASE ---
initialize_database(DB_REPORT_FILE)

# --- 3. CREATE TABS ---
tab1, tab2, tab3 = st.tabs(["Field Survey", "🗺️ Map Visualizer", "📊 Summary"])

# --- 4. TAB 1 (Field Survey) ---
with tab1:
     # --- 1. INITIALIZE SESSION STATE ---
     # We now store a list of results
     if 'analysis_done' not in st.session_state:
          st.session_state.analysis_done = False
          st.session_state.analysis_results_list = []
          st.session_state.overall_details = ""
          
     if 'camera_active' not in st.session_state:
          st.session_state.camera_active = False

     # Function to clear results when a new file is uploaded
     def clear_old_results():
          st.session_state.analysis_done = False
          st.session_state.analysis_results_list = []
          st.session_state.overall_details = ""

     camera_image = None
     image_inputs = []

     # --- 2. CREATE SIDE-BY-SIDE LAYOUT (Unchanged) ---
     col1, col2 = st.columns(2)
     with col1:
          st.markdown("### 📸 Take Photo(s)")
          if st.button("Take Pictures to Upload", 
                         use_container_width=True, 
                         help="Click to activate or deactivate your camera"):
               st.session_state.camera_active = not st.session_state.camera_active

     with col2:
          st.markdown("### 📁 Or Upload Files Manually")
          uploaded_files = st.file_uploader(
               "Drag and drop files here",
               type=["jpg", "jpeg", "png"],
               accept_multiple_files=True,
               label_visibility="collapsed",
               on_change=clear_old_results
          )

     # --- 3. SHOW CAMERA (Unchanged) ---
     if st.session_state.camera_active:
          st.info("Camera is active. You can take one photo at a time.")
          camera_image = st.camera_input(
               "Take a picture of the tree", 
               label_visibility="collapsed",
               on_change=clear_old_results
          )
     
     # --- 4. DECIDE WHICH INPUT TO USE (Unchanged) ---
     if camera_image:
          image_inputs = [camera_image]
          st.session_state.camera_active = False 
     elif uploaded_files:
          image_inputs = uploaded_files
     else:
          if not st.session_state.camera_active:
               st.info("👆 Please take a picture or upload one or more tree images to start analysis.")

     # --- 5. IMAGE PROCESSING & ANALYSIS (LOGIC UPDATED) ---
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

          st.image(images_to_display, caption=image_captions, width=200)
          
          lat, lon = None, None
          lat, lon = get_lat_lon(primary_temp_path)

          if lat and lon:
            st.success(f"✅ Auto-Detected GPS (EXIF): {lat:.6f}, {lon:.6f}")
          else:
               # --- Step 2: Try to get Stamped GPS (OCR) ---
               with st.spinner("No EXIF GPS found. Checking for stamped coordinates (OCR)..."):
                    lat, lon = get_gps_from_stamp(primary_temp_path)
               
               if lat and lon:
                    st.success(f"✅ Auto-Detected GPS (Stamp): {lat:.6f}, {lon:.6f}")
               else:
                    # --- Step 3: Fallback to Manual ---
                    st.warning("⚠️ No auto-detection. Use manual entry:")
                    with st.expander("Enter Coordinates Manually"):
                         manual_lat = st.number_input("Latitude", format="%.6f", value=0.0)
                         manual_lon = st.number_input("Longitude", format="%.6f", value=0.0)
                         if manual_lat != 0.0 or manual_lon != 0.0:
                              lat, lon = manual_lat, manual_lon
                              st.info(f"📍 Using manual coordinates: ({lat}, {lon})")

          # --- "ANALYZE" BUTTON LOGIC (UPDATED) ---
          if st.button("🧠 Analyze Tree Health"):
               with st.spinner("Analyzing all images..."):
                    
                    # --- AI Call now returns a list and details ---
                    results_list, overall_details = analyze_tree_health(temp_file_paths)
                    
                    # --- SAVE TO SESSION STATE ---
                    st.session_state.analysis_results_list = results_list
                    st.session_state.overall_details = overall_details
                    st.session_state.analysis_done = True
                    
                    if not results_list:
                         st.warning(f"Analysis complete, but no specific plants were identified. Details: {overall_details}")

                    # --- SAVE TO DB (Now in a loop) ---
                    timestamp = datetime.datetime.now().isoformat()
                    saved_count = 0
                    for res in results_list:
                         # Get all the data for this specific tree
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
                         "latitude": lat if lat else 0.0,
                         "longitude": lon if lon else 0.0,
                         "details": details,
                         "image_files": ", ".join(image_captions) # Images are same for this batch
                         }
                         saved, message = save_analysis_to_db(DB_REPORT_FILE, data_to_save)
                         if saved:
                              saved_count += 1
                    
                    if saved_count > 0:
                         st.success(f"✅ Analysis complete! {saved_count} tree(s) identified and saved to report.")
                    else:
                         st.error(f"Failed to save any results to report.")

     # --- 6. DISPLAY RESULTS & TREATMENT PLAN (UPDATED) ---
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
               
               # --- Loop through each result and create a card ---
               for i, res in enumerate(results_list):
                    tree_name = res.get('tree_name', 'Unknown')
                    health = res.get('health_condition', 'Unknown')
                    confidence = res.get('confidence_percent', 0)
                    reliability = get_fuzzy_reliability_label(confidence)
                    brief_analysis = res.get('brief_analysis', 'No details')

                    # Create a container for each tree
                    with st.container(border=True):
                         st.subheader(f"Result {i+1}: {tree_name}")
                         
                         col1_res, col2_res, col3_res, col4_res = st.columns(4) 
                         col1_res.metric("🌿 Tree Name", tree_name)
                         col2_res.metric("🩺 Health", health)
                         col3_res.metric("🎯 Confidence", f"{confidence}%")
                         col4_res.metric("📊 Reliability", reliability)
                         
                         st.caption("Brief Analysis")
                         st.markdown(brief_analysis)
                         
                         # --- Treatment plan button FOR EACH tree ---
                         # We use a unique key (f"treat_{i}") so Streamlit knows they are different buttons
                         if st.button(f"👩‍⚕️ Get Treatment Plan for {tree_name}", key=f"treat_{i}"):
                              with st.spinner(f"Generating plan for {tree_name}..."):
                                   plan = get_treatment_plan(tree_name, health, brief_analysis)
                                   st.subheader(f"Recommended Plan for {tree_name}")
                                   st.markdown(plan)

# --- 5. TAB 2 (Map Visualizer) ---
with tab2:
     st.header("🗺️ Tree Health Map")
     
     # Check if the database file exists
     if os.path.exists(DB_REPORT_FILE):
          # 1. Create the map by reading the DB
          health_map = create_health_map(DB_REPORT_FILE)
          
          # 2. Display the map in Streamlit
          st.info("Click on a leaf icon to see the analysis details.")
          st_folium(health_map, use_container_width=True, height=600)
     
     else:
          # Show this message if no report has been saved yet
          st.info("No survey data found. Go to the 'Field Survey' tab to save your first analysis.")

# --- 6. TAB 3 (Summary) ---
with tab3:
     st.header("📊 Survey Summary")

     # Check if the database file exists
     if not os.path.exists(DB_REPORT_FILE):
          st.info("No survey data found. Go to the 'Field Survey' tab to save your first analysis.")
     else:
          # Read data from the database
          conn = sqlite3.connect(DB_REPORT_FILE)
          df = pd.read_sql("SELECT * FROM survey", conn)
          conn.close()
          
          if df.empty:
               st.info("Survey report is empty. Please save some analysis results.")
          else:
               # --- Display Key Metrics ---
               st.subheader("Key Metrics")
               col1_sum, col2_sum, col3_sum = st.columns(3)
               col1_sum.metric("Total Trees Surveyed", len(df))
               try:
                    most_common_issue = df[df['health'] != 'Healthy']['health'].mode()[0]
               except KeyError:
                    most_common_issue = "N/A" # In case no issues are found
               col2_sum.metric("Most Common Issue", most_common_issue)
               col3_sum.metric("Unique Tree Types", df['tree_name'].nunique())
               
               st.markdown("---")
               
               # --- Display Charts ---
               st.subheader("Health Condition Breakdown")
               health_counts = df['health'].value_counts()
               st.bar_chart(health_counts)
               
               st.subheader("Reliability Breakdown")
               reliability_counts = df['reliability'].value_counts()
               st.bar_chart(reliability_counts)
               
               st.markdown("---")
               
               # --- Show Raw Data ---
               st.subheader("Raw Survey Data")
               st.dataframe(df)
               
               # --- delete Data ---
               st.markdown("---")
               st.subheader("⚠️ Danger Zone")
               with st.expander("Clear All Survey Data"):
                    st.warning("This action is permanent and cannot be undone. All survey data will be deleted.")
                         
                    # Add a unique key to this button
                    if st.button("I understand, delete all data", type="primary", key="delete_all_data"):
                         try:
                              conn = sqlite3.connect(DB_REPORT_FILE)
                              cursor = conn.cursor()
                              
                              # SQL command to delete all rows from the survey table
                              cursor.execute("DELETE FROM survey")
                              
                              # Optional: Reset the auto-increment ID
                              cursor.execute("DELETE FROM sqlite_sequence WHERE name='survey'")
                              
                              conn.commit()
                              conn.close()
                              
                              st.success("All survey data has been successfully cleared.")
                              # Rerun the app to refresh the charts and map
                              st.experimental_rerun() 
                              
                         except Exception as e:
                              st.error(f"Error clearing database: {e}")