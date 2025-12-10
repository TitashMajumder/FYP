import streamlit as st
from PIL import Image, ImageDraw
import time
import os
import datetime
import pandas as pd
import sqlite3
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

# --- IMPORT BACKEND MODULES ---
from MapVisualizer import create_health_map
from FuzzyLogic import get_fuzzy_reliability_label
from AIModel import analyze_tree_health, get_treatment_plan, get_gps_from_stamp
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
        padding-bottom: 1rem !important; /* Adjusted as requested */
        margin-bottom: 0.5rem !important; /* Pulls the next element up */
    }
    h3 {
        color: #90EE90 !important;
        padding-top: 0rem !important;    /* Removes top padding */
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
        gap: 3px; /* Space between tabs */
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
initialize_database(DB_REPORT_FILE)
init_training_db()

# --- HELPER FUNCTIONS ---

def save_to_training_db(image_path, tree_name, health_condition):
    """Saves path and labels to training DB."""
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

def draw_diagnosis_box(image_path, box_coords, color="red"):
    """Draws bounding box on image."""
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
            st.markdown("### ⚠️ Database Management")
            
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
    st.stop() # Stop rendering user view

# --- USER VIEW START ---
st.title("🌳 Tree Health Scanner")
st.markdown("### AI-Powered Identification & Diagnosis")

# --- NAVIGATION TABS ---
tab1, tab2 = st.tabs(["📸 Scanner", "🗺️ Map"])

# --- TAB 1: SCANNER ---
with tab1:
    st.write("Upload an image or take a photo. Our system will identify the tree, detect diseases, and suggest cures.")

    # --- INPUT SECTION ---
    if 'camera_active' not in st.session_state: st.session_state.camera_active = False
    if 'manual_lat' not in st.session_state: st.session_state.manual_lat = 0.0
    if 'manual_lon' not in st.session_state: st.session_state.manual_lon = 0.0
    if 'geo_tried' not in st.session_state:
          st.session_state.geo_tried = False
    # State for analysis results (Persistent)
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
    if 'analysis_details' not in st.session_state: st.session_state.analysis_details = ""

    # CALLBACK: Force Reset Coordinates when a new image is uploaded
    def clear_old_results():
          st.session_state.analysis_results = None # Clear old analysis
          st.session_state.analysis_details = ""
          # CRITICAL: Reset these to 0.0 so the GPS logic runs again
          st.session_state.manual_lat = 0.0
          st.session_state.manual_lon = 0.0

    col_cam, col_upl = st.columns(2)
    with col_cam:
        if st.button("📸 Toggle Camera", use_container_width=True):
            st.session_state.camera_active = not st.session_state.camera_active
    with col_upl:
        # ADDED on_change callback here
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

        # 2. AUTOMATIC GPS LOGIC (OCR -> Browser -> Manual)
        if st.session_state.manual_lat == 0.0 and not st.session_state.geo_tried:
            st.session_state.geo_tried = True

            # Step A: Try OCR first
            lat, lon = get_gps_from_stamp(st.session_state.temp_paths[0])
            if lat and lon:
                st.session_state.manual_lat = lat
                st.session_state.manual_lon = lon
                st.success("📍 GPS Found via Image Stamp!")
                st.rerun()

            else:
                # Step B: Try browser GPS once only
                loc = get_geolocation()
                if loc and 'coords' in loc:
                        st.session_state.manual_lat = loc['coords']['latitude']
                        st.session_state.manual_lon = loc['coords']['longitude']
                        st.success("📍 GPS Found via Device!")
                        st.rerun()

        # Step C: Manual Entry (Expander opens if 0.0, otherwise collapsed)
        with st.expander("📍 Coordinates (Auto-detected or Manual)", expanded=(st.session_state.manual_lat == 0.0)):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.manual_lat = st.number_input("Lat", value=st.session_state.manual_lat, format="%.6f")
            with c2:
                st.session_state.manual_lon = st.number_input("Lon", value=st.session_state.manual_lon, format="%.6f")
            
            if st.session_state.manual_lat == 0.0:
                st.caption("⚠️ Could not auto-detect location. Please enter manually.")

        # 3. Analyze Button
        st.write("---")
        
        # --- FIXED: SEPARATE PROCESSING FROM DISPLAY ---
        
        # Only perform analysis if button is clicked
        if st.button("🔍 Analyze Health", type="primary", use_container_width=True):
            status_text = st.empty()
            
            status_text.text("Analyzing...")
            
            # --- CALL REAL AI MODEL ---
            results, details = analyze_tree_health(st.session_state.temp_paths)
            
            # STORE IN SESSION STATE (This fixes the vanishing issue)
            st.session_state.analysis_results = results
            st.session_state.analysis_details = details
            
            status_text.text("Result & Segmented Image...")
            
            # Save to DB (Only once when button is clicked)
            timestamp = datetime.datetime.now().isoformat()
            saved_count = 0
            
            if results:
                for res in results:
                    name = res.get("tree_name", "Unknown")
                    health = res.get("health_condition", "Unknown")
                    confidence = res.get("confidence_percent", 0)
                    reliability = get_fuzzy_reliability_label(confidence)
                    desc = res.get("brief_analysis", "")
                    box = res.get("diseased_area_box")
                    
                    img_idx = res.get('image_index', 0)
                    current_img_path = st.session_state.temp_paths[img_idx] if img_idx < len(st.session_state.temp_paths) else st.session_state.temp_paths[0]
                    
                    segment_path = save_segmented_image(current_img_path, box, name, timestamp)
                    if segment_path:
                        save_to_training_db(segment_path, name, health)
                        
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
                
        # --- DISPLAY RESULTS (Checking Session State) ---
        # This block runs on every rerun, keeping results visible
        if st.session_state.analysis_results:
            st.subheader("📋 Analysis Results")
            results = st.session_state.analysis_results
            
            if not results:
                st.warning("No specific plants identified. Please try a clearer image.")
                st.write(st.session_state.analysis_details)
            
            for i, res in enumerate(results):
                name = res.get("tree_name", "Unknown")
                health = res.get("health_condition", "Unknown")
                confidence = res.get("confidence_percent", 0)
                reliability = get_fuzzy_reliability_label(confidence)
                desc = res.get("brief_analysis", "")
                box = res.get("diseased_area_box")
                if box and len(box) == 4:
                    box = refine_box(box, shrink_ratio=0.18)

                
                img_idx = res.get('image_index', 0)
                if img_idx < len(st.session_state.temp_paths):
                    current_img_path = st.session_state.temp_paths[img_idx]
                else:
                    current_img_path = st.session_state.temp_paths[0] if st.session_state.temp_paths else None

                # --- DISPLAY CARD ---
                color = (0, 255, 0) if health == "Healthy" else (255, 0, 0)

                
                with st.container():
                    st.markdown(f"### {i+1}. {name}")
                    col_img, col_info = st.columns([1.2, 1.8])
                    
                    with col_img:
                        if current_img_path and os.path.exists(current_img_path):
                            if box:
                                annotated = draw_diagnosis_box(current_img_path, box, color)
                                st.image(annotated, caption=f"Visual Diagnosis: {health}", use_container_width=True)
                            else:
                                st.image(current_img_path, caption="Original Image", use_container_width=True)
                        else:
                            st.warning("Image file expired or not found. Please re-upload.")
                    
                    with col_info:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Health", health)
                        m2.metric("Confidence", f"{confidence}%")
                        m3.metric("Reliability", reliability)
                        
                        st.info(f"**Analysis:** {desc}")
                        
                        if st.button(f"💊 Get Cure for {name}", key=f"cure_btn_{i}_{name.replace(' ', '_')}"):
                            with st.spinner("Generating cure..."):
                                plan = get_treatment_plan(name, health, desc)
                                st.success("Treatment Plan Generated:")
                                st.markdown(plan)
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
        # Safe query
        try:
            recents = pd.read_sql("SELECT * FROM survey ORDER BY id DESC LIMIT 5", conn)
        except:
            recents = pd.DataFrame()
        finally:
            conn.close()
        
        if not recents.empty:
            for _, row in recents.iterrows():
                st.markdown(f"**{row['tree_name']}**")
                st.caption(f"{row['timestamp'][:10]} | {row['health']}")
                
                # SAFE ACCESS TO SEGMENT PATH
                if 'segment_path' in recents.columns and pd.notna(row['segment_path']) and os.path.exists(row['segment_path']):
                    st.image(row['segment_path'])
                st.divider()
        else:
            st.write("No history yet.")