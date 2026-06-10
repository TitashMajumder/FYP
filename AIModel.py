# File: AIModel.py
import google.generativeai as genai  # type: ignore
import os
import time
from PIL import Image # type: ignore
import json
import re
import streamlit as st
from dotenv import load_dotenv # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# Global paths for the custom model
CUSTOM_MODEL_PATH = 'plantvillage_tuned_model.h5'
IMAGE_SIZE = 128 
   
# --- Custom Model Prediction Function ---

@st.cache_resource
def get_custom_model():
     import tensorflow as tf  # type: ignore
     from tensorflow.keras.models import load_model  # type: ignore
     with tf.device('/CPU:0'):  # Streamlit cloud doesn't have GPUs
          return load_model(CUSTOM_MODEL_PATH, compile=False)

_class_labels = None
def get_class_labels():
     global _class_labels
     if _class_labels is None:
          with open('class_labels_combined.json', 'r') as f:
               _class_labels = {int(k): v for k, v in json.load(f).items()}
     return _class_labels

_optimal_threshold = None
def get_optimal_threshold():
     global _optimal_threshold
     if _optimal_threshold is None:
          try:
               with open('optimal_threshold.json', 'r') as f:
                    _optimal_threshold = json.load(f)["optimal_threshold"]
          except Exception:
               _optimal_threshold = 0.5  # safe default
     return _optimal_threshold

def load_custom_model_results(image_path):
     """
     Loads and preprocesses an image, then runs the custom CNN model prediction.
     Returns: (predicted_class_name, confidence_percent)
     """
     try:
          cnn_model = get_custom_model()
          
          # Preprocess the image
          img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
          img_array = img_to_array(img)
          img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
          img_array = img_array.astype('float32') / 255.0 # Rescale 
          
          # Predict (verbose=0 suppresses output)
          prediction = cnn_model.predict(img_array, verbose=0)
          
          # Get the top prediction
          threshold = get_optimal_threshold()
          healthy_prob = prediction[0][1]  # index 1 = healthy

          if healthy_prob < threshold:
               predicted_class = "Diseased"
               confidence = 1.0 - healthy_prob
          else:
               predicted_class = "Healthy"
               confidence = healthy_prob
          return predicted_class, float(confidence) * 100.0
          
     except Exception as e:
          print(f"❌ Custom Model Error: {e}")
          # Fallback if the model is not found or fails to load
          return "Model Error", 0.0

# --- 1. CONFIGURE THE model API KEY ---
load_dotenv()
api_key = os.environ.get("model_API_KEY")
if not api_key:
     # Fallback for Streamlit secrets (optional)
     try:
          import streamlit as st # type: ignore
          api_key = st.secrets["model_API_KEY"]
     except:
          pass

if not api_key:
     raise ValueError("model_API_KEY not found. Make sure you have a .env file with the key.")
genai.configure(api_key=api_key)

# --- 2. SET TEMPERATURE TO 0 ---
generation_config = {
     "temperature": 0.0, 
}

model = genai.GenerativeModel(
     'gemini-2.5-flash-lite',
     generation_config=generation_config
)

def is_valid_box(box):
     if not (isinstance(box, (list, tuple)) and len(box) == 4):
          return False
     ymin, xmin, ymax, xmax = box
     return (0 <= ymin < ymax <= 1000) and (0 <= xmin < xmax <= 1000)

def analyze_tree_health(image_paths_list):
     """
     Analyzes images using model, enforcing the 'Healthy', 'Stressed', or 'Diseased' output.
     Uses robust JSON parsing.
     """
     try:
          image_objects = [Image.open(path) for path in image_paths_list]

          # --- REFINED PROMPT FOR AIMODEL.PY ---
          prompt = (
          "You are a plant care and forestry expert. Carefully analyze the provided images. "
          "Focus your inspection on: 1. Leaf health (diseases, spots), 2. Trunk integrity (wounds, "
          "bark loss, mechanical injury, cracks), and 3. Soil conditions (dryness, mold, erosion).\n\n"

          "### HEALTH CATEGORY RULES (STRICT ADHERENCE REQUIRED):\n"
          "- 'Healthy': USE ONLY if the plant shows ABSOLUTELY NO signs of damage, wounds, or disease. "
          "If you see a single crack in the trunk or a dry patch of soil, it is NOT 'Healthy'.\n"
          "- 'Stressed': USE for any abiotic issues: mechanical trunk wounds, bark loss, nutrient deficiency, "
          "or water stress. Mechanical damage is a form of stress.\n"
          "- 'Diseased': USE ONLY for biological infections like fungi, rot, mold, or pests.\n\n"

          "### PRIORITY LOGIC:\n"
          "- If you describe trunk damage, wounds, or missing bark, you MUST set health_condition to 'Stressed'.\n"
          "- NEVER label a wounded tree as 'Healthy'. Failure to follow this rule is a diagnostic error.\n"
          "- If both stress and disease are present, prioritize 'Diseased'.\n\n"

          "### OUTPUT FORMAT:\n"
          "1. Provide a 2-3 line summary of your visual findings.\n"
          "2. Provide a JSON LIST wrapped in a markdown code block (```json ... ```).\n"
          "Each object in the list MUST include:\n"
          "- 'tree_name': Identified species.\n"
          "- 'health_condition': ('Healthy' | 'Stressed' | 'Diseased').\n"
          "- 'confidence_percent': (integer 0-100).\n"
          "- 'brief_analysis': (string) 1-2 sentences explaining the diagnosis.\n"
          "- 'image_index': (integer correlating to the provided images).\n"
          "- 'diseased_area_box': [ymin, xmin, ymax, xmax] (integers 0-1000)\n\n"
          
          "### BOXING RULES:\n"
          "- If the plant is HEALTHY: You MUST return exactly [0, 0, 1000, 1000]. This is mandatory.\n"
          "- If the plant is DISEASED OR STRESSED: Box ONLY the specific affected area (wound, spot, etc.).\n"
          "- 'treatment_plan': (array of 3-5 strings) Step-by-step recovery actions.\n\n"

          "### IMPORTANT:\n"
          "- Identify each unique plant exactly once per image.\n"
          "- Ensure 'treatment_plan' addresses specific trunk wounds if 'Stressed' is selected for damage."
          )

          content_list = [prompt] + image_objects
          response = None
          for attempt in range(3):
               try:
                    response = model.generate_content(content_list)
                    break
               except Exception as e:
                    if attempt == 2:
                         raise
                    print(f"Gemini retry {attempt+1}/3: {e}")
                    time.sleep(2)
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
                    except Exception:
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
                              if not is_valid_box(box):
                                   res['diseased_area_box'] = None

                              treatment = res.get("treatment_plan", [])
                              if not isinstance(treatment, list):
                                   res["treatment_plan"] = []
                                   
                              results_list.append(res)
                    
               except json.JSONDecodeError as e:
                    overall_details = f"Error decoding FINAL JSON list: {e}\nRaw Text: {full_text}"
                    
          else:
               overall_details = full_text
               
          return results_list, overall_details
     # --- ROBUST PARSING END ---
     except Exception as e:
          print(f" Error in model analysis: {str(e)}")
          return [], f"An error occurred: {str(e)}"


def get_gps_from_stamp(image_path):
     """
     Uses model OCR to extract GPS coordinates from image stamps.
     """
     try:
          img = Image.open(image_path)
          prompt = (
               "Analyze this image for any text stamped on it (GPS coordinates). "
               "Return *only* JSON: {\"lat\": float, \"lon\": float}. If none, return the word None."
          )
          ocr_config = {"temperature": 0.0}
          ocr_model = genai.GenerativeModel('gemini-2.5-flash-lite', generation_config=ocr_config)
          response = ocr_model.generate_content([prompt, img])
          text = response.text.strip().replace("```json", "").replace("```", "")
          
          if "none" in text.lower() or "{" not in text: return None, None
          result_json = json.loads(text)
          return float(result_json.get("lat")), float(result_json.get("lon"))
     except Exception as e:
          print(f"GPS extraction failed: {e}")
          return None, None