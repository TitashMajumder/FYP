import streamlit as st # type: ignore
from PIL import Image, ImageDraw # type: ignore
import time
import os
import datetime
import pandas as pd # type: ignore
import sqlite3
from streamlit_folium import st_folium # type: ignore
from streamlit_js_eval import get_geolocation # type: ignore

# --- IMPORT BACKEND MODULES (Updated for Hybrid Fusion) ---
from MapVisualizer import create_health_map
# Importing the new hybrid fusion function
from FuzzyLogic import get_fuzzy_hybrid_analysis 
from AIModel import analyze_tree_health, get_treatment_plan, get_gps_from_stamp, load_custom_model_results 
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
    st.write("Upload an image or take a photo. Our hybrid system fuses custom CNN results with Gemini's detailed analysis for robust diagnosis.")

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
          if not st.session_state.geo_tried:
             lat, lon = get_gps_from_stamp(st.session_state.temp_paths[0])
             if lat and lon:
                 st.session_state.manual_lat = lat
                 st.session_state.manual_lon = lon
                 st.session_state.geo_tried = True # Stop trying
                 st.success("📍 GPS Found via Image Metadata!")
                 st.rerun()

        # Step B: Try Browser GPS if we still don't have coords
        if st.session_state.manual_lat == 0.0:
             loc = get_geolocation()
             
             if loc and 'coords' in loc:
                 st.session_state.manual_lat = loc['coords']['latitude']
                 st.session_state.manual_lon = loc['coords']['longitude']
                 st.session_state.geo_tried = True
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
        st.write("---")
        
        # Only perform analysis if button is clicked
        if st.button("🔍 Analyze Health", type="primary", use_container_width=True):
            status_text = st.empty()
            
            # 1. RUN CUSTOM CNN (Specialized, Binary Check - Only runs on the first image)
            status_text.text("1/3: Custom CNN Analysis (Diseased/Healthy)...")
            first_image_path = st.session_state.temp_paths[0]
            cnn_health, cnn_confidence = load_custom_model_results(first_image_path) 
            
            # 2. RUN GEMINI LLM (Contextual, Multi-species, Detailed Diagnosis)
            status_text.text("2/3: Gemini Contextual Analysis (Species & Bounding Box)...")
            results, details = analyze_tree_health(st.session_state.temp_paths)
            
            # 3. CONSOLIDATE RESULTS AND APPLY FUZZY LOGIC (Decision Fusion)
            status_text.text("3/3: Fusing Decisions & Calculating Reliability...")
            final_results = []

            if results:
                for res in results:
                    gemini_health = res.get("health_condition", "Unknown")
                    gemini_confidence = res.get("confidence_percent", 0)
                    
                    # --- CONDITIONAL CNN DATA CHECK (Fix for Mismatch) ---
                    img_idx = res.get('image_index', 0)
                    
                    if img_idx == 0:
                        current_cnn_health = cnn_health
                        current_cnn_confidence = cnn_confidence
                    else:
                        current_cnn_health = 'N/A'
                        current_cnn_confidence = gemini_confidence 
                    
                    # --- HYBRID CONFIDENCE CALCULATION ---
                    cnn_is_problem = current_cnn_health.lower() == "diseased"
                    gemini_is_problem = gemini_health in ["Diseased", "Stressed"]

                    if gemini_health == "Healthy":
                        confidence_input = 20.0
                    if current_cnn_health == 'N/A' or (cnn_is_problem == gemini_is_problem):
                        confidence_input = max(current_cnn_confidence, gemini_confidence)
                    else:
                        confidence_input = gemini_confidence * 0.8
                        
                    
                    # --- CRITICAL: ADD TRY/EXCEPT BLOCK HERE ---
                    try:
                        # Call the new hybrid function to get the final health decision and reliability
                        final_health_status, reliability_label = get_fuzzy_hybrid_analysis(
                            confidence_input, 
                            gemini_health, 
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
                        st.error(f"Analysis Error: Fuzzy Logic failed for {res.get('tree_name')}. Using raw Gemini data.")
                        
                        # Fallback: Use Gemini's result and assign a low reliability flag
                        res['custom_cnn_confidence'] = current_cnn_confidence
                        res['custom_cnn_health'] = current_cnn_health.capitalize()
                        res['combined_fuzzy_input'] = gemini_confidence
                        res['reliability'] = "Low (Fuzzy Error)"
                        res['health_condition'] = gemini_health
                    
                    final_results.append(res) # Ensure result is appended regardless of fuzzy success
                    
            # STORE FINAL RESULTS in Session State
            st.session_state.analysis_results = final_results
            st.session_state.analysis_details = details
            status_text.text("Analysis Complete.")
            
            # Save to DB 
            timestamp = datetime.datetime.now().isoformat()
            saved_count = 0
            
            if final_results:
                for res in final_results:
                    name = res.get("tree_name", "Unknown")
                    health = res.get("health_condition", "Unknown")
                    confidence = int(res.get("combined_fuzzy_input", 0))
                    reliability = res.get("reliability")
                    desc = res.get("brief_analysis", "")
                    box = res.get("diseased_area_box")
                    
                    img_idx = res.get('image_index', 0)
                    current_img_path = st.session_state.temp_paths[img_idx] if img_idx < len(st.session_state.temp_paths) else st.session_state.temp_paths[0]
                    
                    # Segmentation and Training DB saving
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
                        "segment_path": segment_path # The segment path is now correctly included
                    }
                    save_analysis_to_db(DB_REPORT_FILE, data_to_save) # NOTE: DB schema must be updated
                    saved_count += 1
                
        # --- DISPLAY RESULTS (Checking Session State) ---
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
                    ymin, xmin, ymax, xmax = box
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    min_size = 100 
                    
                    if width > 50 and height > 50:
                        box = refine_box(box, shrink_ratio=0.18) 
                    elif width < min_size or height < min_size:
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        
                        xmin = center_x - (min_size / 2)
                        xmax = center_x + (min_size / 2)
                        ymin = center_y - (min_size / 2)
                        ymax = center_y + (min_size / 2)

                        box = [max(0, int(ymin)), max(0, int(xmin)), min(1000, int(ymax)), min(1000, int(xmax))]
                    
                    if box[0] >= box[2] or box[1] >= box[3]:
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
                    color = "green"  # Use string colors for simplicity during debugging
                elif health == "Stressed":
                    color = "orange" 
                else:
                    color = "red" 
                
                with st.container():
                    st.markdown(f"### {i+1}. {name}")
                    col_img, col_info = st.columns([1.2, 1.8])
                    
                    with col_img:
                        if current_img_path and os.path.exists(current_img_path):
                            # We deliberately skip drawing the box to see the image and data
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
                        m2.metric("Confidence", f"{confidence:.2f}%")
                        m3.metric("Reliability", reliability)
                        st.caption(f"**Custom CNN Check:** {custom_cnn_health} ({custom_cnn_confidence:.2f}%)")
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