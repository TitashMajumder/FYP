# File: AIModel.py
import google.generativeai as genai
import os
from PIL import Image
import json
import re
import io
import base64
from dotenv import load_dotenv

# --- 1. CONFIGURE THE GEMINI API KEY ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
     # Fallback for Streamlit secrets (optional)
     try:
          import streamlit as st
          api_key = st.secrets["GEMINI_API_KEY"]
     except:
          pass

if not api_key:
     raise ValueError("GEMINI_API_KEY not found. Make sure you have a .env file with the key.")
genai.configure(api_key=api_key)

# --- 2. SET TEMPERATURE TO 0 ---
generation_config = {
     "temperature": 0.0, 
}

model = genai.GenerativeModel(
     'gemini-2.5-flash',
     generation_config=generation_config
)

def analyze_tree_health(image_paths_list):
     """
     Analyzes images, identifies DISTINCT plants, asks Gemini for coordinates, 
     and returns deduplicated results.
     
     OPTIMIZED: Removed internal cropping logic since Dashboard handles visualization.
     """
     try:
          image_objects = []
          for path in image_paths_list:
               image_objects.append(Image.open(path))

          # --- PROMPT ---
          prompt = (
               "You are a plant disease expert. Analyze these images. "
               "The images may contain one or MORE different plants/trees. "
               "First, provide a brief overall summary. "
               "Then, provide a JSON LIST. "
               "For EACH DISTINCT plant, identify: 'tree_name', "
               "'health_condition' (Must be exactly one of: 'Healthy', 'Stressed', 'Diseased'), "
               "'confidence_percent', 'brief_analysis', "
               "and crucially: 'image_index' (0, 1, etc.) "
               "and 'diseased_area_box' (bounding box of the diseased area as [ymin, xmin, ymax, xmax] on a scale of 0-1000). "
               "\n\n"
               "IMPORTANT: Do not list the same plant multiple times. Identify each distinct plant exactly once per image."
               "\n\n"
               "Example JSON Item:"
               "{"
               "  'tree_name': 'Mango',"
               "  'health_condition': 'Diseased',"
               "  'confidence_percent': 90,"
               "  'brief_analysis': 'Anthracnose spots visible.',"
               "  'image_index': 0,"
               "  'diseased_area_box': [200, 350, 450, 600]" 
               "}"
          )

          content_list = [prompt] + image_objects
          
          response = model.generate_content(content_list)
          full_text = response.text.strip()
          
          # --- PARSING ---
          json_match = re.search(r"```json\n(\[.*?\])\n```", full_text, re.DOTALL | re.IGNORECASE)
          
          overall_details = "No detailed analysis provided."
          results_list = [] 

          if json_match:
               json_string = json_match.group(1).strip()
               try:
                    raw_results = json.loads(json_string)
               except json.JSONDecodeError:
                    return [], f"Error decoding JSON response: {full_text}"

               overall_details = full_text.split("```json")[0].strip()
               
               # --- DEDUPLICATION LOGIC ---
               seen_plants = set()

               for res in raw_results:
                    idx = res.get('image_index', 0)
                    tree_name = res.get('tree_name', 'Unknown').strip().lower()
                    
                    # Deduplicate primarily by name
                    if tree_name in seen_plants:
                         continue
                    seen_plants.add(tree_name)
                    
                    # Validate the box exists (Dashboard needs this)
                    box = res.get('diseased_area_box', None)
                    if not (box and len(box) == 4):
                         res['diseased_area_box'] = None # Ensure it's None if invalid

                    results_list.append(res)
                    
          else:
               overall_details = full_text
               
          return results_list, overall_details

     except Exception as e:
          print(f"❌ Error in Gemini analysis: {str(e)}")
          return [], f"An error occurred: {str(e)}"

def get_treatment_plan(tree_name, health_condition, analysis_details):
     """
     Generates a treatment plan based on the AI's analysis.
     """
     try:
          prompt = f"""
          You are a plant care expert. Tree: {tree_name}, Condition: {health_condition}, Details: {analysis_details}.
          Provide a simple, actionable, step-by-step treatment plan in markdown bullet points.
          """
          response = model.generate_content(prompt)
          return response.text.strip()
     except Exception as e:
          return "Error: Could not generate treatment plan."

def get_gps_from_stamp(image_path):
     """
     Uses Gemini OCR to extract GPS coordinates from image stamps.
     """
     try:
          img = Image.open(image_path)
          prompt = (
               "Analyze this image for any text stamped on it (GPS coordinates). "
               "Return *only* JSON: {'lat': float, 'lon': float}. If none, return 'None'."
          )
          ocr_config = genai.GenerationConfig(temperature=0.0)
          ocr_model = genai.GenerativeModel('gemini-2.5-flash', generation_config=ocr_config)
          response = ocr_model.generate_content([prompt, img])
          text = response.text.strip().replace("```json", "").replace("```", "")
          
          if "none" in text.lower() or "{" not in text: return None, None
          result_json = json.loads(text)
          return float(result_json.get("lat")), float(result_json.get("lon"))
     except:
          return None, None