# File: AIModel.py
import google.generativeai as genai  # type: ignore
import os
from PIL import Image # type: ignore
import json
import re
import io
import base64
from dotenv import load_dotenv # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
# Use the full path for imports to prevent ambiguity
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# Global paths for the custom model
CUSTOM_MODEL_PATH = 'plantvillage_tuned_model.h5'
IMAGE_SIZE = 128 

# --- Custom Model Prediction Function ---
def load_custom_model_results(image_path):
     """
     Loads and preprocesses an image, then runs the custom CNN model prediction.
     Returns: (predicted_class_name, confidence_percent)
     """
     try:
          # Load model onto the CPU for robust deployment
          with tf.device('/CPU:0'):
               model = load_model(CUSTOM_MODEL_PATH, compile=False) 
               
          # Load class labels saved from the notebook
          with open('class_labels_combined.json', 'r') as f:
               class_labels = json.load(f)
               class_labels = {int(k): v for k, v in class_labels.items()}

          # Preprocess the image
          img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
          img_array = img_to_array(img)
          img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
          img_array = img_array.astype('float32') / 255.0 # Rescale 
          
          # Predict (verbose=0 suppresses output)
          prediction = model.predict(img_array, verbose=0)
          
          # Get the top prediction
          predicted_index = np.argmax(prediction[0])
          confidence = prediction[0][predicted_index]
          predicted_class = class_labels.get(predicted_index, "Unknown")
          
          return predicted_class, float(confidence) * 100.0
          
     except Exception as e:
          print(f"❌ Custom Model Error: {e}")
          # Fallback if the model is not found or fails to load
          return "Model Error", 0.0

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
     'gemini-2.5-flash-lite',
     generation_config=generation_config
)

def analyze_tree_health(image_paths_list):
     """
     Analyzes images using Gemini, enforcing the 'Healthy', 'Stressed', or 'Diseased' output.
     Uses robust JSON parsing.
     """
     try:
          image_objects = [Image.open(path) for path in image_paths_list]

          # --- PROMPT ---
          prompt = (
               "You are a plant disease expert. Analyze these images. "
               "The images may contain one or MORE different plants/trees. "
               "First, provide a brief overall summary. "
               "Then, provide a JSON LIST. "
               "For EACH DISTINCT plant, identify: 'tree_name', "
               "'health_condition' (Must be exactly one of: 'Healthy', 'Stressed', or 'Diseased'). "
               "Use 'Stressed' for issues caused by abiotic factors (water, light, nutrients) or very early, mild infections. "
               "PRIORITY RULE: If the analysis mentions nutrient deficiency, water stress, or environmental stress, you MUST output 'Stressed' for the health_condition, even if the image could suggest disease."
               "'confidence_percent', 'brief_analysis', "
               "and crucially: 'image_index' (0, 1, etc.) "
               "and 'diseased_area_box' (bounding box of the diseased area as [ymin, xmin, ymax, xmax] on a scale of 0-1000). "
               "\n\n"
               "BOXING RULES:"
               "- If the plant is HEALTHY: The box must cover the ENTIRE TREE/PLANT, with coordinates close to [0, 0, 1000, 1000]."
               "- If the plant is DISEASED OR STRESSED: The box must cover ONLY the specific affected area (e.g., spot, patch of chlorosis)."
               "\n\n"
               "IMPORTANT: Do not list the same plant multiple times. Identify each distinct plant exactly once per image."
               "\n\n"
               "Example JSON Item: {'tree_name': 'Mango', 'health_condition': 'Diseased', 'confidence_percent': 90, 'brief_analysis': 'Anthracnose spots visible.', 'image_index': 0, 'diseased_area_box': [200, 350, 450, 600]}"
          )

          content_list = [prompt] + image_objects
          
          response = model.generate_content(content_list)
          full_text = response.text.strip()
          
          # --- ROBUST JSON PARSING (FIXED) ---
          json_match = re.search(r"```json\s*(\[.*?\])\s*```", full_text, re.DOTALL | re.IGNORECASE)
          
          json_string = None
          overall_details = full_text.split("```json")[0].strip() if "```json" in full_text else full_text
          results_list = []

          if json_match:
               json_string = json_match.group(1).strip()
          else:
               # Fallback: Search for an array structure directly (Handles missing code fences)
               array_match = re.search(r"(\[.*?\])", full_text, re.DOTALL)
               if array_match:
                    json_string = array_match.group(1).strip()
                    # Try to separate overall details if array found
                    try:
                         pre_text = full_text.split(json_string)[0].strip()
                         if len(pre_text) > 10: overall_details = pre_text
                    except:
                         pass

          if json_string:
               # Clean up common LLM errors (e.g., single quotes, trailing commas)
               json_string = json_string.replace("'", '"')
               json_string = re.sub(r',\s*\]', ']', json_string)
               
               try:
                    raw_results = json.loads(json_string)
                    
                    seen_plants = set()
                    for res in raw_results:
                         if isinstance(res, dict) and res.get('tree_name'):
                              tree_name = res.get('tree_name', 'Unknown').strip().lower()
                              
                              if tree_name in seen_plants: continue
                              seen_plants.add(tree_name)
                              
                              box = res.get('diseased_area_box', None)
                              if not (isinstance(box, (list, tuple)) and len(box) == 4): 
                                   res['diseased_area_box'] = None

                              results_list.append(res)
                    
               except json.JSONDecodeError as e:
                    overall_details = f"Error decoding FINAL JSON list: {e}\nRaw Text: {full_text}"
                    
          else:
               overall_details = full_text
               
          return results_list, overall_details
     # --- ROBUST PARSING END ---

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