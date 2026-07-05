import streamlit as st # type: ignore
from PIL import Image, ImageDraw # type: ignore
import time
import os
import datetime
import pandas as pd # type: ignore
import sqlite3
from fpdf import FPDF # type: ignore
import hashlib
import numpy as np
import glob
from matplotlib import cm
from streamlit_folium import st_folium # type: ignore
from streamlit_js_eval import get_geolocation # type: ignore

# --- IMPORT BACKEND MODULES ---
from MapVisualizer import create_health_map
from FuzzyLogic import get_fuzzy_hybrid_analysis 
from AIModel import analyze_tree_health, get_gps_from_stamp, load_custom_model_results, generate_disease_heatmap
from ReportGenerator import initialize_database, save_analysis_to_db
from init_db import init_training_db
from config import DB_REPORT_FILE as _DB_REPORT, DB_TRAINING_FILE as _DB_TRAINING  # Fix #19

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Help the Greens 🌿", page_icon="🌳", layout="wide")

def load_css(path):
     with open(path, encoding="utf-8") as f:
          st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("assets/styles.css")  # Fix #7: removed incorrect assets/ prefix

# --- DATABASE SETUP --- (Fix #19: paths now from config.py)
DB_REPORT_FILE   = _DB_REPORT
DB_TRAINING_FILE = _DB_TRAINING

# Initialize DBs on app load
@st.cache_resource
def setup_databases():
    initialize_database(DB_REPORT_FILE)
    init_training_db()

setup_databases()

def pdf_safe(text):
     """Fix #7: FPDF's core fonts only support Latin-1. Replace any
     characters outside that range (emojis, special unicode, etc.)
     instead of letting pdf.cell()/multi_cell() throw."""
     return str(text).encode('latin-1', errors='replace').decode('latin-1')

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
          pdf.cell(60, 10, pdf_safe(row['tree_name'])[:25], 1)
          pdf.cell(40, 10, pdf_safe(row['health']), 1)
          pdf.cell(40, 10, pdf_safe(row['reliability']), 1)
          pdf.cell(50, 10, f"{row['latitude']:.2f}, {row['longitude']:.2f}", 1)
          pdf.ln()
     
     # Use dest='S' which returns a bytearray in fpdf2; encode safely (Fix #6)
     raw = pdf.output(dest='S')
     if isinstance(raw, (bytes, bytearray)):
          return bytes(raw)
     return raw.encode('latin-1', errors='replace')

def export_single_result_pdf(name, health, confidence, reliability, treatment_plan, analysis, current_img_path=None, heatmap_img=None):
     """
     Creates a detailed PDF report for a single tree diagnosis.
     Used for individual result export from scanner.
     """
     pdf = FPDF()
     pdf.add_page()
     
     # Title
     pdf.set_font("Arial", 'B', 16)
     pdf.cell(200, 10, f"Tree Diagnosis Report", ln=True, align='C')
     pdf.ln(5)
     
     # Tree Info Section
     pdf.set_font("Arial", 'B', 14)
     pdf.cell(200, 10, f"Species: {pdf_safe(name)}", ln=True)
     
     pdf.set_font("Arial", '', 11)
     pdf.cell(100, 8, f"Health Status: {pdf_safe(health)}")
     pdf.cell(100, 8, f"Confidence: {confidence:.1f}%", ln=True)
     pdf.cell(100, 8, f"Reliability: {pdf_safe(reliability)}", ln=True)
     
     pdf.ln(5)
     
     # Add Images Section
     pdf.set_font("Arial", 'B', 12)
     pdf.cell(200, 10, "Images", ln=True)
     
     try:
          # Add diagnosis box image
          if current_img_path and os.path.exists(current_img_path):
               pdf.image(current_img_path, x=10, y=pdf.get_y(), w=90)
               pdf.set_y(pdf.get_y() + 55)
     except Exception as e:
          print(f"Error adding diagnosis image to PDF: {e}")
     
     try:
          # Add heatmap image if available
          if heatmap_img is not None:
               # Save heatmap temporarily
               temp_heatmap_path = "temp_heatmap.png"
               heatmap_img.save(temp_heatmap_path)
               pdf.image(temp_heatmap_path, x=110, y=pdf.get_y() - 55, w=90)
               pdf.set_y(pdf.get_y() + 55)
               os.remove(temp_heatmap_path)
     except Exception as e:
          print(f"Error adding heatmap image to PDF: {e}")
     
     pdf.ln(5)
     
     # Analysis Section
     pdf.set_font("Arial", 'B', 12)
     pdf.cell(200, 10, "Detailed Analysis", ln=True)
     
     pdf.set_font("Arial", '', 10)
     pdf.multi_cell(200, 5, pdf_safe(analysis))
     
     pdf.ln(5)
     
     # Treatment Plan Section
     pdf.set_font("Arial", 'B', 12)
     pdf.cell(200, 10, "Treatment Plan", ln=True)
     
     pdf.set_font("Arial", '', 10)
     for i, step in enumerate(treatment_plan, 1):
          pdf.multi_cell(200, 5, pdf_safe(f"{i}. {step}"))
     
     pdf.ln(5)
     
     # Footer
     pdf.set_font("Arial", 'I', 8)
     pdf.cell(200, 10, "Generated by Help the Greens - AI Tree Health Monitor", 
               ln=True, align='C')
     
     # Use dest='S' which returns a bytearray in fpdf2; encode safely (Fix #6)
     raw = pdf.output(dest='S')
     if isinstance(raw, (bytes, bytearray)):
          return bytes(raw)
     return raw.encode('latin-1', errors='replace')

def overlay_heatmap_on_image(original_image_path, heatmap_array):
    """
    Overlay Grad-CAM heatmap on original image.
    Returns PIL Image with overlay.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Load original image
        img = Image.open(original_image_path).convert('RGB')
        original_size = img.size  # (width, height)
        
        # Normalize heatmap to 0-1 range
        heatmap_normalized = (heatmap_array - heatmap_array.min()) / (heatmap_array.max() - heatmap_array.min() + 1e-5)
        
        # Apply colormap (hot = red for disease)
        cmap = plt.get_cmap('hot')
        heatmap_colored = cmap(heatmap_normalized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_colored)
        
        # Resize heatmap to match original image size
        heatmap_img = heatmap_img.resize(original_size, Image.LANCZOS)
        
        # Blend with original (40% heatmap, 60% original)
        blended = Image.blend(img, heatmap_img, alpha=0.4)
        return blended
        
    except Exception as e:
        print(f"Heatmap overlay error: {e}")
        return Image.open(original_image_path)
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
               st.dataframe(df, width='stretch')
               
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
                         width='stretch'
                    )

               with col_down2:
                    try:
                         pdf_data = create_pdf_report(df)
                         st.download_button(
                              label="📄 Download PDF (Field Report)",
                              data=pdf_data,
                              file_name="Tree_Health_Summary.pdf",
                              mime="application/pdf",
                              width='stretch'
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
page = st.sidebar.radio("NAVIGATION", ["🔍 User Scanner", "🔐 Admin Dashboard"], horizontal=False)

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
     if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

     def clear_old_results():
          st.session_state.analysis_results = None
          st.session_state.analysis_details = ""
          st.session_state.manual_lat = 0.0
          st.session_state.manual_lon = 0.0
          st.session_state.geo_tried = False 
          st.session_state.results_saved = False

     # Native uniform grid setup
     col_cam, col_upl = st.columns(2, gap="medium")
     
     with col_cam:
          if st.button("📸 Toggle Camera", key="toggle_camera", width='stretch'):
               st.session_state.camera_active = not st.session_state.camera_active

     # --- FIX: Initialize uploaded_files safely at the top scope ---
     uploaded_files = []

     with col_upl:
          has_images = st.session_state.get('temp_paths') and len(st.session_state.temp_paths) > 0
          
          if has_images:
               # Render the uniform Remove button matching the layout
               if st.button("❌ Remove Uploaded Image", key="clear_files_btn", width='stretch'):
                    st.session_state.temp_paths.clear()
                    st.session_state.uploader_key += 1
                    clear_old_results()
                    st.rerun()
               
               # Uploader kept alive for state retention; styled via CSS .hidden-uploader
               with st.container():
                    uploaded_files = st.file_uploader(
                         "📁 Upload Images", type=['jpg','png','jpeg'], accept_multiple_files=True,
                         label_visibility="collapsed", on_change=clear_old_results, key=f"file_uploader_{st.session_state.uploader_key}"
                    )
          else:
               # Normal empty state display
               uploaded_files = st.file_uploader(
                    "📁 Upload Images", type=['jpg','png','jpeg'], accept_multiple_files=True,
                    label_visibility="collapsed", on_change=clear_old_results, key=f"file_uploader_{st.session_state.uploader_key}"
               )

     camera_image = None
     if st.session_state.camera_active:
          camera_image = st.camera_input("Capture Image", label_visibility="collapsed")
          if camera_image:
               clear_old_results()

     image_inputs = []
     if camera_image: image_inputs.append(camera_image)
     if uploaded_files: image_inputs.extend(uploaded_files)

     # ===== SHOW LANDING PAGE ONLY IF NO IMAGES UPLOADED =====
     if not image_inputs:
          st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(144,238,144,0.3), transparent); margin: 20px 0;"></div>', unsafe_allow_html=True)
          
          # 1. QUICK STATS - ENHANCED Uniform Classes
          st.markdown('<p class="section-heading">📊 OVERVIEW</p>', unsafe_allow_html=True)
          
          col1, col2, col3, col4 = st.columns(4, gap="medium")
          
          if os.path.exists(DB_REPORT_FILE):
               conn = sqlite3.connect(DB_REPORT_FILE)
               try:
                    total = pd.read_sql("SELECT COUNT(*) as count FROM survey", conn).iloc[0]['count']
                    diseased = pd.read_sql("SELECT COUNT(*) as count FROM survey WHERE health='Diseased'", conn).iloc[0]['count']
                    healthy = pd.read_sql("SELECT COUNT(*) as count FROM survey WHERE health='Healthy'", conn).iloc[0]['count']
                    conn.close()
                    
                    with col1:
                         st.markdown(f"""
                         <div style="background: linear-gradient(135deg, rgba(30,58,48,0.8) 0%, rgba(46,139,87,0.3) 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(144,238,144,0.2); text-align: center;">
                              <p style="font-size: 12px; color: #888; margin: 0; text-transform: uppercase;">Total Scans</p>
                              <p style="font-size: 32px; color: #90EE90; font-weight: bold; margin: 5px 0 0 0;">{total}</p>
                         </div>
                         """, unsafe_allow_html=True)
                    
                    with col2:
                         st.markdown(f"""
                         <div style="background: linear-gradient(135deg, rgba(80,30,30,0.8) 0%, rgba(220,38,38,0.2) 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(220,38,38,0.2); text-align: center;">
                              <p style="font-size: 12px; color: #888; margin: 0; text-transform: uppercase;">Diseased</p>
                              <p style="font-size: 32px; color: #ff6b6b; font-weight: bold; margin: 5px 0 0 0;">{diseased}</p>
                         </div>
                         """, unsafe_allow_html=True)
                    
                    with col3:
                         st.markdown(f"""
                         <div style="background: linear-gradient(135deg, rgba(30,60,40,0.8) 0%, rgba(34,197,94,0.2) 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(34,197,94,0.2); text-align: center;">
                              <p style="font-size: 12px; color: #888; margin: 0; text-transform: uppercase;">Healthy</p>
                              <p style="font-size: 32px; color: #22c55e; font-weight: bold; margin: 5px 0 0 0;">{healthy}</p>
                         </div>
                         """, unsafe_allow_html=True)
                    
                    with col4:
                         accuracy = f"{(healthy/(total+1)*100):.0f}%" if total > 0 else "0%"
                         st.markdown(f"""
                         <div style="background: linear-gradient(135deg, rgba(30,50,70,0.8) 0%, rgba(59,130,246,0.2) 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(59,130,246,0.2); text-align: center;">
                              <p style="font-size: 12px; color: #888; margin: 0; text-transform: uppercase;">Health Rate</p>
                              <p style="font-size: 32px; color: #3b82f6; font-weight: bold; margin: 5px 0 0 0;">{accuracy}</p>
                         </div>
                         """, unsafe_allow_html=True)
               except Exception:
                    pass
          
          st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(144,238,144,0.3), transparent); margin: 20px 0;"></div>', unsafe_allow_html=True)
          
          # 2. FEATURE HIGHLIGHTS - Standardized via Column Architecture
          st.markdown('<p class="section-heading">📊 FEATURES</p>', unsafe_allow_html=True)
          
          f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
          with f_col1:
               st.markdown("""
               <div class="feature-card" style="background: linear-gradient(135deg, rgba(46,139,87,0.2) 0%, rgba(46,139,87,0.05) 100%); border-color: rgba(144,238,144,0.25);">
                  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                       <div style="font-size: 32px;">🤖</div>
                       <h4 style="margin: 0;">AI-Powered</h4>
                  </div>
                  <p style="margin: 0; color: #b9c2c9; font-size: 13px;">Hybrid CNN + model Vision for accurate diagnosis</p>
               </div>
               """, unsafe_allow_html=True)
               
          with f_col2:
               st.markdown("""
               <div class="feature-card" style="background: linear-gradient(135deg, rgba(244,114,182,0.2) 0%, rgba(244,114,182,0.05) 100%); border-color: rgba(244,114,182,0.25);">
                  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                       <div style="font-size: 32px;">📍</div>
                       <h4 style="color: #f472b6 !important; margin: 0;">Location Tracking</h4>
                  </div>
                  <p style="margin: 0; color: #b9c2c9; font-size: 13px;">Auto GPS detection & navigation links included</p>
               </div>
               """, unsafe_allow_html=True)
               
          with f_col3:
               st.markdown("""
               <div class="feature-card" style="background: linear-gradient(135deg, rgba(99,102,241,0.2) 0%, rgba(99,102,241,0.05) 100%); border-color: rgba(99,102,241,0.25);">
                  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                       <div style="font-size: 32px;">📊</div>
                       <h4 style="color: #6366f1 !important; margin: 0;">Visual Analytics</h4>
                  </div>
                  <p style="margin: 0; color: #b9c2c9; font-size: 13px;">Disease heatmaps & detailed PDF reports</p>
               </div>
               """, unsafe_allow_html=True)
          
          st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(144,238,144,0.3), transparent); margin: 20px 0;"></div>', unsafe_allow_html=True)
          
          # 3. WORKFLOW - Standardized alignment architecture matching the Features layout
          st.markdown('<p class="section-heading">🎓 WORKFLOW</p>', unsafe_allow_html=True)
          
          workflow_cols = st.columns(5, gap="medium")
          workflows = [
               {"emoji": "📸", "title": "Upload", "desc": "Image or Photo"},
               {"emoji": "🧠", "title": "Analyze", "desc": "Dual AI"},
               {"emoji": "⚙️", "title": "Fuse", "desc": "Logic"},
               {"emoji": "💊", "title": "Treat", "desc": "Plan"},
               {"emoji": "📄", "title": "Report", "desc": "Download"}
          ]
          
          for col, workflow in zip(workflow_cols, workflows):
               with col:
                    st.markdown(f"""
                    <div class="feature-card" style="background: linear-gradient(135deg, rgba(46,139,87,0.15) 0%, rgba(46,139,87,0.03) 100%); border-color: rgba(144,238,144,0.2); text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                         <div style="font-size: 36px; margin-bottom: 10px;">{workflow['emoji']}</div>
                         <p style="margin: 0; color: #90EE90; font-weight: bold; font-size: 14px;">{workflow['title']}</p>
                         <p style="margin: 5px 0 0 0; color: #888; font-size: 12px;">{workflow['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
          
          st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(144,238,144,0.3), transparent); margin: 20px 0;"></div>', unsafe_allow_html=True)
          
          # 4. TIPS FOR BEST RESULTS
          st.markdown('<p class="section-heading">💡 TIPS FOR BEST RESULTS</p>', unsafe_allow_html=True)
          col_do, col_dont = st.columns(2, gap="medium")
          
          with col_do:
               st.markdown("""
               <div style="background: linear-gradient(135deg, rgba(34,197,94,0.12) 0%, rgba(34,197,94,0.02) 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(34,197,94,0.25);">
               <h4 style="color: #22c55e !important; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">✅ DO:</h4>
               <ul style="margin: 0; padding-left: 20px; color: #b9c2c9;">
                    <li style="margin-bottom: 8px;">Natural daylight photos</li>
                    <li style="margin-bottom: 8px;">Show full plant shape</li>
                    <li style="margin-bottom: 8px;">Include leaves & branches</li>
                    <li style="margin-bottom: 8px;">High-resolution camera</li>
                    <li style="margin-bottom: 0;">Multiple angles</li>
               </ul>
               </div>
               """, unsafe_allow_html=True)
          
          with col_dont:
               st.markdown("""
               <div style="background: linear-gradient(135deg, rgba(220,38,38,0.12) 0%, rgba(220,38,38,0.02) 100%); padding: 20px; border-radius: 12px; border: 1px solid rgba(220,38,38,0.25);">
               <h4 style="color: #dc2626 !important; margin: 0 0 15px 0; display: flex; align-items: center; gap: 8px;">❌ DON'T:</h4>
               <ul style="margin: 0; padding-left: 20px; color: #b9c2c9;">
                    <li style="margin-bottom: 8px;">Blurry/low light photos</li>
                    <li style="margin-bottom: 8px;">Single leaf close-ups</li>
                    <li style="margin-bottom: 8px;">Photos too far away</li>
                    <li style="margin-bottom: 8px;">Extreme angles</li>
                    <li style="margin-bottom: 0;">Shadows or glare</li>
               </ul>
               </div>
               """, unsafe_allow_html=True)
               
     # ===== END: LANDING PAGE HIDDEN WHEN IMAGES UPLOAD =====
     if image_inputs:
          # 1. Save inputs to temp — Fix #8: only rewrite/clean temp files
          # when the uploaded content actually changed, not on every rerun
          # (e.g. typing into the lat/lon fields used to leak a fresh
          # scan_*.jpg on every keystroke).
          os.makedirs("temp", exist_ok=True)
          if 'temp_paths' not in st.session_state:
               st.session_state.temp_paths = []
          if 'last_upload_hash' not in st.session_state:
               st.session_state.last_upload_hash = None

          incoming_hash = hashlib.md5(
               b"".join(f.getbuffer() for f in image_inputs)
          ).hexdigest()
          is_new_upload = st.session_state.last_upload_hash != incoming_hash

          if is_new_upload:
               # Remove the previous batch before writing the new one
               for old_file in glob.glob(os.path.join("temp", "scan_*.jpg")):
                    try:
                         os.remove(old_file)
                    except Exception:
                         pass

               st.session_state.temp_paths = []
               for i, img_file in enumerate(image_inputs):
                    fname = f"scan_{int(time.time())}_{i}.jpg"
                    fpath = os.path.join("temp", fname)
                    with open(fpath, "wb") as f:
                         f.write(img_file.getbuffer())
                    st.session_state.temp_paths.append(fpath)

               st.session_state.last_upload_hash = incoming_hash

          # Display thumbnails (always — uses whatever temp_paths currently holds)
          cols = st.columns(len(st.session_state.temp_paths)) if len(st.session_state.temp_paths) < 4 else st.columns(4)
          for i, fpath in enumerate(st.session_state.temp_paths):
               with cols[i % 4]:
                    st.image(fpath, width=100)

          # 🔥 NEW IMAGE DETECTED (ONCE) — reuses the upload-hash check above
          if is_new_upload:
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

          # Step C: Manual Entry (Stationary Bordered Container)
          with st.container(border=True):
               st.markdown("**📍 Coordinates (Auto-detected or Manual)**")
               c1, c2 = st.columns(2)
               with c1:
                    st.session_state.manual_lat = st.number_input("Lat", value=st.session_state.manual_lat, format="%.6f")
               with c2:
                    st.session_state.manual_lon = st.number_input("Lon", value=st.session_state.manual_lon, format="%.6f")
               
               if st.session_state.manual_lat == 0.0:
                    st.caption("⚠️ Could not auto-detect location. Please enter manually.")

          # 3. Analyze Button
          st.markdown("""
               <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(144,238,144,0.3), transparent); margin: 20px 0;"></div>
               """, unsafe_allow_html=True)
        
          # Only perform analysis if button is clicked
          if st.button("🔍 Analyze Health", type="primary", width='stretch'):
               st.session_state.critical_alerts = []
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
                              res['original_health_condition'] = model_health
                              res['health_condition'] = final_health_status
                         
                         except Exception as e:
                              print(f"FUZZY FUSION FAILED for {res.get('tree_name', 'Unknown')}: {e}")
                              st.error(f"Analysis Error: Fuzzy Logic failed for {res.get('tree_name')}. Using raw model data.")
                              
                              # Fallback: Use model model's result and assign a low reliability flag
                              res['custom_cnn_confidence'] = current_cnn_confidence
                              res['custom_cnn_health'] = current_cnn_health.capitalize()
                              res['combined_fuzzy_input'] = model_confidence
                              res['reliability'] = "Low (Fuzzy Error)"
                              res['original_health_condition'] = model_health
                              res['health_condition'] = model_health
                         
                         final_results.append(res)
                    
               # STORE FINAL RESULTS in Session State
               st.session_state.analysis_results = final_results
               st.session_state.analysis_details = details
               time.sleep(1.5)
               status_text.empty()

               # Clean up stale temp files AFTER analysis is complete (Fix #5)
               current_files = set(os.path.abspath(f) for f in st.session_state.temp_paths)
               for old_file in glob.glob(os.path.join("temp", "scan_*.jpg")):
                    if os.path.abspath(old_file) not in current_files:
                         try:
                              os.remove(old_file)
                         except Exception:
                              pass
               
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
                         original_health = res.get("original_health_condition", health)
                         box = res.get("diseased_area_box") if original_health != "Healthy" else None

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
                    
          if st.session_state.analysis_results is not None:
               results = st.session_state.analysis_results
               if not results:
                    st.markdown('<h3 style="margin-bottom: 20px;">📋 Analysis Results</h3>', unsafe_allow_html=True)
                    st.warning("No specific plants identified. Please try a clearer image.")
                    st.write(st.session_state.analysis_details)
               else:
                    st.markdown('<h3 style="margin-bottom: 20px;">📋 Analysis Results</h3>', unsafe_allow_html=True)
                    
                    for i, res in enumerate(results):
                         name = res.get("tree_name", "Unknown")
                         health = res.get("health_condition", "Unknown")
                         confidence = res.get("combined_fuzzy_input", 0)
                         reliability = res.get("reliability")
                         custom_cnn_confidence = res.get('custom_cnn_confidence', 0)
                         custom_cnn_health = res.get('custom_cnn_health', 'N/A')
                         desc = res.get("brief_analysis", "")
                         box = res.get("diseased_area_box")
                         original_health = res.get("original_health_condition", health)
                    
                         if box and len(box) == 4 and original_health != "Healthy":
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

                         if health=="Healthy":
                                   badge="status-healthy"
                         elif health=="Stressed":
                                   badge="status-stressed"
                         else:
                                   badge="status-diseased"

                         st.markdown(f"""
                                             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                             <h2 style="margin: 0; color: #90EE90;">🌳 {name}</h2>
                                             <span class="status-{health.lower()}" style="margin: 0;">{health.upper()}</span>
                                             </div>
                                             """, unsafe_allow_html=True)
                         col_img, col_info = st.columns([1.2, 2.5])
                         with col_img:
                              if current_img_path and os.path.exists(current_img_path):
                                   # Generate heatmap ONCE for both display and PDF
                                   heatmap_img_for_pdf = None
                                   if health in ["Diseased", "Stressed"]:
                                        with st.spinner("Generating heatmap..."):
                                             heatmap, diseased_prob = generate_disease_heatmap(current_img_path)
                                             if heatmap is not None:
                                                  heatmap_img_for_pdf = overlay_heatmap_on_image(current_img_path, heatmap)
                                   
                                   # Display diagnosis box and heatmap SIDE BY SIDE
                                   img_col1, img_col2 = st.columns(2)
                                   
                                   with img_col1:
                                        if box:
                                             label_text = f"{health} ({reliability})"
                                             annotated = draw_diagnosis_box(current_img_path, box, color, label_text)
                                             if annotated is not None and isinstance(annotated, Image.Image):
                                                  st.image(annotated, caption="Diagnosis Box", width='stretch')
                                        else:
                                             # Healthy trees (or no box): show the original image
                                             st.image(current_img_path, caption="Original Image", width='stretch')
                              
                                   with img_col2:
                                        if heatmap_img_for_pdf is not None:
                                             st.image(heatmap_img_for_pdf, caption=f"Severity ({diseased_prob*100:.1f}%)", width='stretch')
                              else:
                                   st.warning("Image not found")
                         
                         with col_info:
                                   m1,m2,m3 = st.columns(3)

                                   with m1:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                        <div class="metric-title">🌿 Health</div>
                                        <div class="metric-value">{health}</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                   with m2:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                        <div class="metric-title">🎯 Confidence</div>
                                        <div class="metric-value">{res.get('confidence_percent', confidence):.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                   with m3:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                        <div class="metric-title">⭐ Reliability</div>
                                        <div class="metric-value">{reliability}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                   st.markdown(
                                        f"""
                                        <div class="analysis-card">
                                        🧠Diagnosis: 
                                        {desc}
                                        </div>
                                        """,unsafe_allow_html=True)
                                   
                                   # NEW — PDF EXPORT BUTTON
                                   pdf_data = export_single_result_pdf(
                                        name, 
                                        health, 
                                        confidence, 
                                        reliability,
                                        res.get("treatment_plan", []),
                                        desc,
                                        current_img_path,
                                        heatmap_img_for_pdf
                                   )
                                   # Navigation to tree location
                                   # 3 Action Buttons: Cure, Maps, Export
                                   st.markdown("")
                                   action_col1, action_col2, action_col3 = st.columns(3, gap="medium")
                                   
                                   # Get Cure Button (already exists, move to col1)
                                   with action_col1:
                                        if st.button(f"💊 Get Cure for {name}", key=f"cure_btn_{i}_{name.replace(' ', '_')}", width='stretch'):
                                             with st.spinner("Generating cure..."):
                                                  time.sleep(0.6)
                                                  treatment_plan = res.get("treatment_plan", [])
                                                  if treatment_plan:
                                                       st.success("Treatment Plan")
                                                       for step in treatment_plan:
                                                            st.markdown(f"- {step}")
                                                  else:
                                                       st.info("No treatment required. The plant is healthy.")
                                   
                                   # Google Maps Button
                                   with action_col2:
                                        if st.session_state.manual_lat != 0.0 and st.session_state.manual_lon != 0.0:
                                             maps_url = f"https://www.google.com/maps/search/{st.session_state.manual_lat},{st.session_state.manual_lon}"
                                             st.markdown(f"""
                                             <a href="{maps_url}" target="_blank" style="text-decoration: none;">
                                             <button style="width: 100%; padding: 12px 20px; background: linear-gradient(135deg, #2E8B57 0%, #1e6d3b 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 15px; height: 48px; display: flex; align-items: center; justify-content: center;">
                                             🗺️ Google Maps
                                             </button>
                                             </a>
                                             """, unsafe_allow_html=True)
                                        else:
                                             st.warning("📍 GPS not available")
                                   
                                   # Export PDF Button
                                   with action_col3:
                                        st.download_button(
                                             label=f"📄 Export Report",
                                             data=pdf_data,
                                             file_name=f"{name}_diagnosis_{int(time.time())}.pdf",
                                             mime="application/pdf",
                                             key=f"pdf_btn_{i}_{name.replace(' ', '_')}",
                                             width='stretch'
                                        )

# --- TAB 2: MAP ---
with tab2:
     st.header("🌍 Tree Health Map & Locality Heatmap")
     
     if os.path.exists(DB_REPORT_FILE):
          try:
               from LocalityHeatmap import create_disease_heatmap, create_health_distribution_chart
               from streamlit_folium import st_folium
               
               # Check if we have data with GPS coordinates
               conn = sqlite3.connect(DB_REPORT_FILE)
               check_query = "SELECT COUNT(*) as count FROM survey WHERE latitude != 0 AND longitude != 0"
               data_count = pd.read_sql(check_query, conn).iloc[0]['count']
               conn.close()
               
               if data_count == 0:
                    st.warning("⚠️ No GPS data available yet. Please scan trees with GPS enabled.")
                    st.info("💡 Tip: Allow location access when uploading images to enable map features.")
               else:
                    map_tab1, map_tab2, map_tab3 = st.tabs(["🗺️ Standard Map", "🔥 Disease Heatmap", "📊 Locality Stats"])
                    
                    with map_tab1:
                         st.subheader("Individual Tree Locations")
                         try:
                              map_obj = create_health_map(DB_REPORT_FILE)
                              st_folium(map_obj, height=500, use_container_width=True, key="standard_map")
                         except Exception as e:
                              st.error(f"Error loading standard map: {e}")
                    
                    with map_tab2:
                         st.subheader("Disease Concentration Heatmap")
                         st.write("🟢 Green = Healthy | 🟡 Yellow = Medium | 🔴 Red = High Disease")
                         try:
                              heatmap = create_disease_heatmap(DB_REPORT_FILE)
                              if heatmap:
                                   st.components.v1.html(heatmap._repr_html_(), height=600, scrolling=False)
                              else:
                                   st.warning("Could not generate heatmap. Check database data.")
                         except Exception as e:
                              st.error(f"Error loading disease heatmap: {e}")
                                        
                    with map_tab3:
                         st.subheader("Locality-wise Health Analysis")
                         try:
                              create_health_distribution_chart(DB_REPORT_FILE)
                         except Exception as e:
                              st.error(f"Error loading locality stats: {e}")
          
          except ImportError:
               st.error("❌ LocalityHeatmap module not found!")
               st.info("""
               **Fix:**
               1. Make sure `LocalityHeatmap.py` is in the same folder as `dashboard.py`
               2. Install folium: `pip install folium`
               3. Restart the app
               """)
     
     else:
          st.info("No database found. Please scan some trees first.")

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