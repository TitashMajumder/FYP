import streamlit as st # type: ignore
from PIL import Image, ImageDraw # type: ignore
import time
import os
import datetime
import pandas as pd # type: ignore
import sqlite3
from fpdf import FPDF # type: ignore
import hashlib
import glob
from streamlit_folium import st_folium # type: ignore
from streamlit_js_eval import get_geolocation # type: ignore

# --- IMPORT BACKEND MODULES ---
from MapVisualizer import create_health_map
from FuzzyLogic import get_fuzzy_hybrid_analysis 
from AIModel import analyze_tree_health, get_gps_from_stamp, load_custom_model_results 
from ReportGenerator import initialize_database, save_analysis_to_db
from init_db import init_training_db

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Help the Greens 🌿", page_icon="🌳", layout="wide")

st.markdown("""
    <style>
    /* Reduce top padding for the main container */
    .block-container {
        padding-top: 2.5rem; 
        padding-bottom: 1rem;
    }

    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #102b1f 100%);
        color: white;
        font-family: 'Trebuchet MS', sans-serif;
    }
    
    /* --- HEADINGS GAP FIX --- */
    h1 {
        color: #90EE90 !important;
        padding-bottom: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        color: #90EE90 !important;
        padding-top: 0rem !important;
    }
    h2 {
        color: #90EE90 !important;
    }

    /* --- BUTTON STYLING --- */
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        min-height: 68px;
        font-size: 1.2rem;
        font-weight: 600;
        white-space: normal; 
        word-wrap: break-word;
    }
    
    .stButton>button:hover {
        background-color: #3CB371;
        transform: scale(1.02);
    }
    
    div[data-testid="stMetricValue"] {
        color: #90EE90;
    }
    
    /* --- TAB STYLING --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        border-bottom: 2px solid #2E8B57;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto; 
        min-width: auto; 
        white-space: nowrap; 
        background-color: #1E3A2F;
        border-radius: 8px 8px 0px 0px;
        padding: 12px 20px; 
        color: #cfcfcf;
        font-size: 1.2rem; 
        font-weight: 600;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57 !important;
        color: white !important;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #90EE90;
        border-color: #90EE90;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATABASE SETUP ---
DB_REPORT_FILE = "tree_survey.db"
DB_TRAINING_FILE = "training_dataset.db"

# Initialize DBs on app load
@st.cache_resource
def setup_databases():
    initialize_database(DB_REPORT_FILE)
    init_training_db()

setup_databases()

# --- HELPER FUNCTIONS ---
def get_image_hash(image_path):
     with open(image_path, "rb") as f:
          return hashlib.md5(f.read()).hexdigest()

def save_to_training_db(image_path, tree_name, health_condition):
     """Saves path and labels to training DB."""
     conn = None
     try:
          conn = sqlite3.connect(DB_TRAINING_FILE)
          cursor = conn.cursor()
          timestamp = datetime.datetime.now().isoformat()
          cursor.execute("INSERT INTO training_data (timestamp, image_path, label_tree_name, label_health_condition) VALUES (?, ?, ?, ?)", 
                         (timestamp, image_path, tree_name, health_condition))
          conn.commit()
     except Exception as e:
          print(f"Error saving to training DB: {e}")
     finally:
          if conn:
               conn.close()

def refine_box(box, shrink_ratio=0.15):
     """
     Shrinks a bounding box to remove background noise.
     box = [ymin, xmin, ymax, xmax] in 0–1000 scale
     """
     ymin, xmin, ymax, xmax = box

     width = xmax - xmin
     height = ymax - ymin

     xmin = xmin + shrink_ratio * width
     xmax = xmax - shrink_ratio * width
     ymin = ymin + shrink_ratio * height
     ymax = ymax - shrink_ratio * height

     return [int(ymin), int(xmin), int(ymax), int(xmax)]

def save_segmented_image(original_image_path, box_coords, tree_name, timestamp_str):
     """Crops and saves the segmented diseased area."""
     try:
          if not box_coords or len(box_coords) != 4: return None
          img = Image.open(original_image_path)
          w, h = img.size
          ymin, xmin, ymax, xmax = box_coords
          
          left = (xmin / 1000) * w
          top = (ymin / 1000) * h
          right = (xmax / 1000) * w
          bottom = (ymax / 1000) * h
          
          cropped_img = img.crop((left, top, right, bottom))
          
          save_dir = "segments"
          os.makedirs(save_dir, exist_ok=True)
          
          clean_name = "".join(c for c in tree_name if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
          clean_time = timestamp_str.replace(':', '').replace('-', '').replace('.', '')
          filename = f"{clean_name}_{clean_time}.jpg"
          file_path = os.path.join(save_dir, filename)
          
          cropped_img.save(file_path)
          return file_path
     except Exception as e:
          print(f"Error saving segment: {e}")
          return None

def draw_diagnosis_box(image_path, box_coords, color="red", label=""):
     """Draws bounding box on image, scaled from 0-1000 coordinates."""
     try:
          img = Image.open(image_path)
          if img.mode != "RGB": img = img.convert("RGB")
          draw = ImageDraw.Draw(img)
          w, h = img.size
          
          if box_coords and len(box_coords) == 4:
               ymin, xmin, ymax, xmax = box_coords
               left = (xmin / 1000) * w
               top = (ymin / 1000) * h
               right = (xmax / 1000) * w
               bottom = (ymax / 1000) * h
               draw.rectangle([left, top, right, bottom], outline=color, width=8)
          return img
     except Exception as e:
          return Image.open(image_path)
     
def create_pdf_report(df):
     pdf = FPDF()
     pdf.add_page()
     pdf.set_font("Arial", 'B', 16)
     pdf.cell(200, 10, txt="Help the Greens: Official Health Report", ln=True, align='C')
     pdf.ln(10)
     
     # Summary Table Headers
     pdf.set_font("Arial", 'B', 12)
     pdf.cell(60, 10, "Tree Name", 1)
     pdf.cell(40, 10, "Health", 1)
     pdf.cell(40, 10, "Reliability", 1)
     pdf.cell(50, 10, "Location", 1)
     pdf.ln()
     
     # Table Content
     pdf.set_font("Arial", size=10)
     for _, row in df.iterrows():
          pdf.cell(60, 10, str(row['tree_name'])[:25], 1)
          pdf.cell(40, 10, str(row['health']), 1)
          pdf.cell(40, 10, str(row['reliability']), 1)
          pdf.cell(50, 10, f"{row['latitude']:.2f}, {row['longitude']:.2f}", 1)
          pdf.ln()
     
     return pdf.output(dest='S').encode('latin-1')

# -------------------------
# FEATURE: ADMIN DASHBOARD
# -------------------------
def admin_dashboard():
     st.title("📊 Admin Dashboard & Analytics")
     
     if os.path.exists(DB_REPORT_FILE):
          conn = sqlite3.connect(DB_REPORT_FILE)
          try:
               df = pd.read_sql("SELECT * FROM survey ORDER BY id DESC", conn)
          except Exception as e:
               st.error(f"Database schema error: {e}")
               st.warning("You may need to reset the database.")
               df = pd.DataFrame()
          finally:
               conn.close()
          
          if not df.empty:
               # --- ADMIN ALERTS ---
               # Filter for high-confidence diseased trees
               alerts = df[(df['health'] == 'Diseased') & (df['reliability'] == 'High')]
               if not alerts.empty:
                    st.error(f"⚠️ **Action Required:** {len(alerts)} trees are in critical condition with high diagnostic reliability.")
                    with st.expander("🔍 View Critical Tree List"):
                         for _, row in alerts.iterrows():
                              st.write(f"- **{row['tree_name']}** (ID: {row['id']}) at Lat: {row['latitude']}, Lon: {row['longitude']}")
                              
               # Metrics
               c1, c2, c3 = st.columns(3)
               c1.metric("Total Surveys", len(df))
               c2.metric("Diseased Trees", len(df[df['health'] == 'Diseased']))
               c3.metric("Unique Species", df['tree_name'].nunique())
               
               st.divider()
            
               # Charts
               col_chart1, col_chart2 = st.columns(2)
               with col_chart1:
                    st.subheader("Health Distribution")
                    st.bar_chart(df['health'].value_counts())
               with col_chart2:
                    st.subheader("Reliability Scores")
                    st.bar_chart(df['reliability'].value_counts())
                    
               st.subheader("Raw Data")
               st.dataframe(df, use_container_width=True)
               
               # Delete Button
               st.markdown("### Database Management")
            
               col_del1, col_del2 = st.columns(2)
               with col_del1:
                    if st.button("🗑️ Clear Records (Keep Schema)"):
                         conn = sqlite3.connect(DB_REPORT_FILE)
                         conn.execute("DELETE FROM survey")
                         conn.commit()
                         conn.close()
                         st.rerun()
               
               with col_del2:
                    if st.button("🔄 Reset Database (Fix Schema Error)"):
                         try:
                              if os.path.exists(DB_REPORT_FILE):
                                   os.remove(DB_REPORT_FILE)
                              initialize_database(DB_REPORT_FILE)
                              st.success("Database reset successfully! The error should be gone.")
                              time.sleep(2)
                              st.rerun()
                         except Exception as e:
                              st.error(f"Error resetting DB: {e}")

               st.subheader("📥 Download Reports")
               col_down1, col_down2 = st.columns(2)

               with col_down1:
                    st.download_button(
                         label="📊 Download CSV ",
                         data=df.to_csv(index=False).encode('utf-8'),
                         file_name='tree_health_data.csv',
                         mime='text/csv',
                         use_container_width=True
                    )

               with col_down2:
                    try:
                         pdf_data = create_pdf_report(df)
                         st.download_button(
                              label="📄 Download PDF (Field Report)",
                              data=pdf_data,
                              file_name="Tree_Health_Summary.pdf",
                              mime="application/pdf",
                              use_container_width=True
                         )
                    except Exception as e:
                         st.error(f"PDF Error: {e}")
          else:
               st.info("Database is empty.")
     else:
          st.info("Database file not found.")

# -------------------------
# MAIN APP LOGIC
# -------------------------

# --- Sidebar ---
st.sidebar.title("🌿 Help the Greens")
page = st.sidebar.selectbox("Navigation", ["🔍 User Scanner", "🔐 Admin Dashboard"])

# --- NEW: ABOUT SECTION ---
with st.sidebar.expander("ℹ️ About", expanded=False):
    st.info("AI-Powered Tree Health Monitor")
    st.markdown("""
    **How to use:**
    1. Go to **Scanner** tab.
    2. Upload an image or take a photo.
    3. Click **Analyze Health**.
    4. View results & get treatment plans!
    
    *Built for a greener future.* 🌍
    """)

if page == "🔐 Admin Dashboard":
    admin_dashboard()
    st.stop()

# --- USER VIEW START ---
st.title("🌳 Tree Health Scanner")
st.markdown("### AI-Powered Identification & Diagnosis")

# --- NAVIGATION TABS ---
tab1, tab2 = st.tabs(["📸 Scanner", "🗺️ Map"])

# --- TAB 1: SCANNER ---
with tab1:
     st.write("Upload an image or take a photo. Our hybrid system fuses custom CNN results with model's detailed analysis for robust diagnosis.")

     # --- INPUT SECTION ---
     if 'camera_active' not in st.session_state: st.session_state.camera_active = False
     if 'manual_lat' not in st.session_state: st.session_state.manual_lat = 0.0
     if 'manual_lon' not in st.session_state: st.session_state.manual_lon = 0.0
     if 'geo_tried' not in st.session_state: st.session_state.geo_tried = False
     if "gps_bound_to_image" not in st.session_state: st.session_state.gps_bound_to_image = False
     if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
     if 'analysis_details' not in st.session_state: st.session_state.analysis_details = ""
     if 'results_saved' not in st.session_state: st.session_state.results_saved = False
     if "critical_alerts" not in st.session_state: st.session_state.critical_alerts = []
     
     # CALLBACK: Force Reset Coordinates when a new image is uploaded
     def clear_old_results():
          st.session_state.analysis_results = None
          st.session_state.analysis_details = ""
          st.session_state.manual_lat = 0.0
          st.session_state.manual_lon = 0.0
          st.session_state.geo_tried = False 
          st.session_state.results_saved = False

     col_cam, col_upl = st.columns(2)
     with col_cam:
          if st.button("📸 Toggle Camera", use_container_width=True):
               st.session_state.camera_active = not st.session_state.camera_active
     with col_upl:
          uploaded_files = st.file_uploader("📁 Upload Images", type=['jpg','png','jpeg'], accept_multiple_files=True, label_visibility="collapsed", on_change=clear_old_results)

     camera_image = None
     if st.session_state.camera_active:
          camera_image = st.camera_input("Capture Image", label_visibility="collapsed")
          if camera_image:
                    clear_old_results()

     # Gather Inputs
     image_inputs = []
     if camera_image: image_inputs.append(camera_image)
     if uploaded_files: image_inputs.extend(uploaded_files)

     if image_inputs:
          # 1. Save inputs to temp
          os.makedirs("temp", exist_ok=True)
          if 'temp_paths' not in st.session_state:
                    st.session_state.temp_paths = []
          else:
                    st.session_state.temp_paths.clear()
          
          # Display thumbnails
          cols = st.columns(len(image_inputs)) if len(image_inputs) < 4 else st.columns(4)
          
          for i, img_file in enumerate(image_inputs):
               fname = f"scan_{int(time.time())}_{i}.jpg"
               fpath = os.path.join("temp", fname)
               with open(fpath, "wb") as f:
                    f.write(img_file.getbuffer())
               st.session_state.temp_paths.append(fpath)
               
               # Show small thumbnail
               with cols[i % 4]:
                    st.image(fpath, width=100)
          
          current_image_hash = get_image_hash(st.session_state.temp_paths[0])

          if "last_image_hash" not in st.session_state:
               st.session_state.last_image_hash = None

          # 🔥 NEW IMAGE DETECTED (ONCE)
          if st.session_state.last_image_hash != current_image_hash:
               st.session_state.last_image_hash = current_image_hash
               # Reset GPS ONLY once per new image
               st.session_state.manual_lat = 0.0
               st.session_state.manual_lon = 0.0
               st.session_state.geo_tried = False
               st.session_state.gps_bound_to_image = False
               st.session_state.critical_alerts = []


          # 2. AUTOMATIC GPS LOGIC (OCR -> Browser -> Manual)
          if (st.session_state.manual_lat == 0.0 and not st.session_state.geo_tried):
               lat, lon = get_gps_from_stamp(st.session_state.temp_paths[0])
               st.session_state.geo_tried = True

               # Step A: Try OCR first
               if lat and lon:
                    st.session_state.manual_lat = lat
                    st.session_state.manual_lon = lon
                    st.session_state.geo_tried = True 
                    st.session_state.gps_bound_to_image = True
                    st.success("📍 GPS Found via Image Metadata!")
                    st.rerun()
               else:
                    # Mark that stamp attempt is done, but GPS not found
                    st.session_state.geo_tried = "stamp_failed"

          # Step B: Try Browser GPS if we still don't have coords
          if (st.session_state.manual_lat == 0.0 and st.session_state.geo_tried == "stamp_failed"):
               loc = get_geolocation()
               
               if loc and 'coords' in loc:
                    st.session_state.manual_lat = loc['coords']['latitude']
                    st.session_state.manual_lon = loc['coords']['longitude']
                    st.session_state.geo_tried = True
                    st.session_state.gps_bound_to_image = True
                    st.success("📍 GPS Found via Device!")
                    st.rerun()

          # Step C: Manual Entry 
          with st.expander("📍 Coordinates (Auto-detected or Manual)", expanded=(st.session_state.manual_lat == 0.0)):
               c1, c2 = st.columns(2)
               with c1:
                    st.session_state.manual_lat = st.number_input("Lat", value=st.session_state.manual_lat, format="%.6f")
               with c2:
                    st.session_state.manual_lon = st.number_input("Lon", value=st.session_state.manual_lon, format="%.6f")
               
               if st.session_state.manual_lat == 0.0:
                    st.caption("⚠️ Could not auto-detect location. Please enter manually.")

          # 3. Analyze Button
          st.divider() 
        
          # Only perform analysis if button is clicked
          if st.button("🔍 Analyze Health", type="primary", use_container_width=True):
               st.session_state.critical_alerts = []
               current_files = set(os.path.abspath(f) for f in st.session_state.temp_paths)
               for old_file in glob.glob(os.path.join("temp", "scan_*.jpg")):
                    if os.path.abspath(old_file) not in current_files:
                         try:
                              os.remove(old_file)
                         except Exception:
                              pass
               status_text = st.empty()
               
               # 1. RUN CUSTOM CNN (Specialized, Binary Check - Only runs on the first image)
               status_text.text("1/3: Custom CNN Analysis (Diseased/Healthy)...")
               cnn_results = {}
               for idx, img_path in enumerate(st.session_state.temp_paths):
                    health, confidence = load_custom_model_results(img_path)
                    cnn_results[idx] = {
                         "health": health,
                         "confidence": confidence
                    }
               
               # 2. RUN model LLM (Contextual, Multi-species, Detailed Diagnosis)
               status_text.text("2/3: model Contextual Analysis (Species & Bounding Box)...")
               results, details = analyze_tree_health(st.session_state.temp_paths)
               
               # 3. CONSOLIDATE RESULTS AND APPLY FUZZY LOGIC (Decision Fusion)
               status_text.text("3/3: Fusing Decisions & Calculating Reliability...")
               final_results = []

               if results:
                    for res in results:
                         model_health = res.get("health_condition", "Unknown")
                         if model_health not in ["Healthy", "Stressed", "Diseased"]:
                              model_health = "Stressed"
                         model_confidence = res.get("confidence_percent", 0)
                         
                         # --- CONDITIONAL CNN DATA CHECK (Fix for Mismatch) ---
                         img_idx = res.get('image_index', 0)
                         
                         cnn_data = cnn_results.get(img_idx)

                         if cnn_data:
                              current_cnn_health = cnn_data["health"]
                              current_cnn_confidence = cnn_data["confidence"]
                         else:
                              current_cnn_health = "N/A"
                              current_cnn_confidence = model_confidence 
                              
                         # --- HYBRID CONFIDENCE CALCULATION ---
                         cnn_is_problem = current_cnn_health.lower() == "diseased"
                         model_is_problem = model_health in ["Diseased", "Stressed"]

                         # FIX — use elif
                         if model_health == "Healthy":
                              confidence_input = 20.0
                         elif current_cnn_health == 'N/A' or (cnn_is_problem == model_is_problem):
                              confidence_input = max(current_cnn_confidence, model_confidence)
                         else:
                              confidence_input = model_confidence * 0.8
                         confidence_input = float(confidence_input)
                         
                         try:
                              # Call the new hybrid function to get the final health decision and reliability
                              final_health_status, reliability_label = get_fuzzy_hybrid_analysis(
                                   confidence_input, 
                                   model_health, 
                                   current_cnn_health
                              )
                              
                              # Use the final calculated score for display
                              comparison_confidence = confidence_input 

                              # Augment the result dictionary
                              res['custom_cnn_confidence'] = current_cnn_confidence
                              res['custom_cnn_health'] = current_cnn_health.capitalize()
                              res['combined_fuzzy_input'] = comparison_confidence
                              res['reliability'] = reliability_label         
                              res['health_condition'] = final_health_status 
                         
                         except Exception as e:
                              print(f"FUZZY FUSION FAILED for {res.get('tree_name', 'Unknown')}: {e}")
                              st.error(f"Analysis Error: Fuzzy Logic failed for {res.get('tree_name')}. Using raw model data.")
                              
                              # Fallback: Use model's result and assign a low reliability flag
                              res['custom_cnn_confidence'] = current_cnn_confidence
                              res['custom_cnn_health'] = current_cnn_health.capitalize()
                              res['combined_fuzzy_input'] = model_confidence
                              res['reliability'] = "Low (Fuzzy Error)"
                              res['health_condition'] = model_health
                         
                         final_results.append(res)
                    
               # STORE FINAL RESULTS in Session State
               st.session_state.analysis_results = final_results
               st.session_state.analysis_details = details
               time.sleep(1.5)
               status_text.empty()
               
               # --- ALERT NOTIFICATION HERE ---
               for res in final_results:
                    if res.get("health_condition") == "Diseased" and res.get("reliability") == "High":
                         st.session_state.critical_alerts.append(
                              f"🚨 High-Risk Tree: {res.get('tree_name')}"
                         )
                         
               # Save to DB (RUN ONLY ONCE PER ANALYSIS)
               if final_results and not st.session_state.results_saved:
                    timestamp = datetime.datetime.now().isoformat()
                    saved_count = 0

                    for res in final_results:
                         name = res.get("tree_name", "Unknown")
                         health = res.get("health_condition", "Unknown")
                         confidence = int(res.get("combined_fuzzy_input", 0))
                         reliability = res.get("reliability")
                         desc = res.get("brief_analysis", "")
                         box = res.get("diseased_area_box")

                         img_idx = res.get("image_index", 0)
                         current_img_path = (
                              st.session_state.temp_paths[img_idx]
                              if img_idx < len(st.session_state.temp_paths)
                              else st.session_state.temp_paths[0]
                         )

                         # Segmentation and Training DB saving
                         segment_path = save_segmented_image(
                              current_img_path, box, name, timestamp
                         )

                         # SAVE TO TRAINING DB ONLY AFTER FINAL DECISION
                         if health in ["Healthy", "Diseased", "Stressed"]:
                              # Save full image for Healthy
                              if health == "Healthy":
                                   save_to_training_db(
                                        current_img_path,
                                        name,
                                        health
                                   )

                              # Save cropped segment for Diseased / Stressed
                              else:
                                   if segment_path:
                                        save_to_training_db(
                                             segment_path,
                                             name,
                                             health
                                        )

                         data_to_save = {
                              "timestamp": timestamp,
                              "tree_name": name,
                              "health": health,
                              "confidence": confidence,
                              "reliability": reliability,
                              "latitude": st.session_state.manual_lat,
                              "longitude": st.session_state.manual_lon,
                              "details": desc,
                              "image_files": os.path.basename(current_img_path),
                              "segment_path": segment_path
                         }

                         save_analysis_to_db(DB_REPORT_FILE, data_to_save)
                         saved_count += 1

                    st.session_state.results_saved = True
                    st.toast(f"✅ {saved_count} result(s) saved successfully.")
     
          # --- DISPLAY RESULTS (Checking Session State) ---
          if "critical_alerts" in st.session_state and st.session_state.critical_alerts:
               st.error("⚠️ CRITICAL ALERTS")
               for msg in st.session_state.critical_alerts:
                    st.markdown(f"- {msg}")
                    
          if st.session_state.analysis_results:
               st.subheader("📋 Analysis Results")
               results = st.session_state.analysis_results
               
               if not results:
                    st.warning("No specific plants identified. Please try a clearer image.")
                    st.write(st.session_state.analysis_details)
               
               for i, res in enumerate(results):
                    name = res.get("tree_name", "Unknown")
                    health = res.get("health_condition", "Unknown")
                    confidence = res.get("combined_fuzzy_input", 0)
                    reliability = res.get("reliability")
                    custom_cnn_confidence = res.get('custom_cnn_confidence', 0)
                    custom_cnn_health = res.get('custom_cnn_health', 'N/A')
                    desc = res.get("brief_analysis", "")
                    box = res.get("diseased_area_box")
                    
                    if box and len(box) == 4:
                         if health == "Healthy":
                              box = None
                         else:
                              ymin, xmin, ymax, xmax = box
                              # Clamp values
                              ymin = max(0, ymin)
                              xmin = max(0, xmin)
                              ymax = min(1000, ymax)
                              xmax = min(1000, xmax)

                              # Validate box
                              if ymax > ymin and xmax > xmin:
                                        box = refine_box([ymin, xmin, ymax, xmax], shrink_ratio=0.18)
                              else:
                                   box = None
                    else:
                         box = None
                         
                    img_idx = res.get('image_index', 0)
                    if img_idx < len(st.session_state.temp_paths):
                         current_img_path = st.session_state.temp_paths[img_idx]
                    else:
                         current_img_path = st.session_state.temp_paths[0] if st.session_state.temp_paths else None

                    # --- DISPLAY CARD (3-Class Color Logic) ---
                    if health == "Healthy":
                         color = "green"
                    elif health == "Stressed":
                         color = "orange" 
                    else:
                         color = "red" 
                    
                    with st.container():
                         st.markdown(f"### {i+1}. {name}")
                         col_img, col_info = st.columns([0.8, 2.2])
                         
                         with col_img:
                              if current_img_path and os.path.exists(current_img_path):
                                   if box:
                                        label_text = f"{health} ({reliability})"
                                        annotated = draw_diagnosis_box(current_img_path, box, color, label_text)
                                        st.image(annotated, caption=f"Visual Diagnosis: {health}", use_container_width=True)
                                   else:
                                        st.image(current_img_path, caption="Original Image", use_container_width=True)
                              else:
                                   st.warning("Image file expired or not found. Please re-upload.")
                         
                         with col_info:
                              m1, m2, m3 = st.columns(3)
                              m1.metric("Health", health)
                              m2.metric("Confidence", f"{confidence:.2f}%")
                              m3.metric("Reliability", reliability)
                              st.caption(f"**Custom CNN Check:** {custom_cnn_health} ({custom_cnn_confidence:.2f}%)")
                              st.info(f"**Analysis:** {desc}")
                              
                              if st.button(f"💊 Get Cure for {name}", key=f"cure_btn_{i}_{name.replace(' ', '_')}"):
                                   with st.spinner("Generating cure..."):
                                        time.sleep(0.6)
                                        treatment_plan = res.get("treatment_plan", [])
                                        if treatment_plan:
                                             st.success("Treatment Plan")
                                             for step in treatment_plan:
                                                  st.markdown(f"- {step}")
                                        else:
                                             st.info("No treatment required. The plant is healthy.")
                         st.divider()

# --- TAB 2: MAP ---
with tab2:
     st.header("🌍 Tree Health Map")
     st.write("View the location and health status of all surveyed trees.")
     if os.path.exists(DB_REPORT_FILE):
          map_obj = create_health_map(DB_REPORT_FILE)
          st_folium(map_obj, width=None, height=500, use_container_width=True)
     else:
          st.info("No data available for the map yet. Please scan some trees first.")

# --- SIDEBAR HISTORY (Real DB) ---
with st.sidebar.expander("📚 Recent Scans", expanded=True):
     if os.path.exists(DB_REPORT_FILE):
          conn = sqlite3.connect(DB_REPORT_FILE)
          try:
               recents = pd.read_sql("SELECT * FROM survey ORDER BY id DESC LIMIT 5", conn)
          except Exception:
               recents = pd.DataFrame()
          finally:
               conn.close()
          
          if not recents.empty:
               for _, row in recents.iterrows():
                    st.markdown(f"**{row['tree_name']}**")
                    st.caption(f"{row['timestamp'][:10]} | {row['health']}")
                    
                    if 'segment_path' in recents.columns and pd.notna(row['segment_path']) and os.path.exists(row['segment_path']):
                         st.image(row['segment_path'])
                    st.divider()
          else:
               st.write("No history yet.")