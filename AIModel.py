# File: AIModel.py
import google.generativeai as genai  # type: ignore
import os
import time
from PIL import Image # type: ignore
import json
import re
import streamlit as st
import matplotlib
matplotlib.use('Agg')
from dotenv import load_dotenv # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# Global paths for the custom model
CUSTOM_MODEL_PATH = 'models/plantvillage_tuned_model copy.h5'
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
          with open('models/class_labels_combined.json', 'r') as f:
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
          return "Unknown", 0.0  # Fix #9: consistent unknown label instead of "Model Error"

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
    "response_mime_type": "application/json"
}

model = genai.GenerativeModel(
    "gemini-2.5-flash-lite",
    generation_config=generation_config
)

def is_valid_box(box, health_condition=None):
     """Validates a bounding box. Fix #11: full-image box [0,0,1000,1000] is
     only acceptable for Healthy trees; for diseased/stressed it indicates
     the model failed to localise the lesion and should be rejected."""
     if not (isinstance(box, (list, tuple)) and len(box) == 4):
          return False
     ymin, xmin, ymax, xmax = box
     if not ((0 <= ymin < ymax <= 1000) and (0 <= xmin < xmax <= 1000)):
          return False
     # Reject trivially full-image boxes for non-healthy trees
     if health_condition and health_condition != 'Healthy':
          if ymin == 0 and xmin == 0 and ymax == 1000 and xmax == 1000:
               return False
     return True

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
                    response = model.generate_content(
                    content_list,
                    generation_config={
                         "temperature": 0.0,
                         "response_mime_type": "application/json"
                         }
                    )
                    break
               except Exception as e:
                    if attempt == 2:
                         raise
                    print(f"model retry {attempt+1}/3: {e}")
                    time.sleep(2)
          
          results_list = []

          try:
               raw_results = json.loads(response.text)

               seen_plants = set()

               for res in raw_results:

                    if not isinstance(res, dict):
                         continue

                    tree_name = res.get("tree_name", "Unknown").strip().lower()
                    img_idx = res.get("image_index", 0)

                    dedup_key = (img_idx, tree_name)

                    if dedup_key in seen_plants:
                         continue

                    seen_plants.add(dedup_key)

                    box = res.get("diseased_area_box")

                    if not is_valid_box(box, res.get("health_condition")):
                         res["diseased_area_box"] = None

                    if not isinstance(res.get("treatment_plan"), list):
                         res["treatment_plan"] = []

                    results_list.append(res)

               return results_list, ""

          except json.JSONDecodeError as e:

               return [], f"Gemini returned invalid JSON:\n\n{e}\n\n{response.text}"
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

def generate_disease_heatmap(image_path, model=None):
     """
     Generate Grad-CAM heatmap showing where CNN detects disease.
     Returns the heatmap array overlaid on original image.
     """
     try:
          if model is None:
               model = get_custom_model()
          
          # Load and preprocess image
          img = Image.open(image_path).convert('RGB')
          original_img = np.array(img)
          
          img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
          img_array = np.array(img_resized, dtype='float32') / 255.0
          img_array = np.expand_dims(img_array, axis=0)
          
          # Get predictions
          predictions = model.predict(img_array, verbose=0)
          diseased_prob = predictions[0][0]  # index 0 = diseased
          
          # Fix #10: validate layer name exists before building gradient model
          layer_names = [l.name for l in model.layers]
          target_layer = 'conv2d_7'
          if target_layer not in layer_names:
               # Fall back to last Conv2D layer found
               conv_layers = [l.name for l in model.layers if 'conv2d' in l.name]
               if not conv_layers:
                    print(f'No conv2d layers found in model. Available: {layer_names}')
                    return None, 0.0
               target_layer = conv_layers[-1]
               print(f'Layer conv2d_7 not found; using {target_layer} instead')
          # Build gradient model
          grad_model = tf.keras.Model(
               inputs=model.input,
               outputs=[
                    model.get_layer(name=target_layer).output,  # last conv layer
                    model.output
               ]
          )
          
          # Compute gradients
          with tf.GradientTape() as tape:
               conv_outputs, preds = grad_model(img_array)
               diseased_class_channel = preds[:, 0]
          
          grads = tape.gradient(diseased_class_channel, conv_outputs)
          pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
          
          # Generate heatmap
          conv_outputs = conv_outputs[0]
          heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
          heatmap = tf.squeeze(heatmap)
          heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-5)
          heatmap_resized = tf.image.resize(
               tf.expand_dims(heatmap, 0), 
               (IMAGE_SIZE, IMAGE_SIZE)
          )[0].numpy()
          
          # Scale to original image size
          heatmap_scaled = tf.image.resize(
               tf.expand_dims(heatmap_resized, 0),
               (original_img.shape[0], original_img.shape[1])
          )[0].numpy()
          
          return heatmap_scaled, diseased_prob
          
     except Exception as e:
          print(f"Grad-CAM error: {e}")
          return None, 0.0
     
def generate_weather_advice_ai(weather, diagnosis, treatment_plan=None):
     """Gemini-backed weather advisory. Kept for optional use (e.g. richer,
     diagnosis-aware phrasing) but NOT called by default — see
     generate_weather_advice() below, which covers the same three sections
     deterministically without spending an API call on every single scan."""
     plan_text = "\n".join(f"- {step}" for step in (treatment_plan or [])) or "Not available."
     prompt = f"""
     You are a professional arborist.

     Current Weather:
     - Temperature: {weather['temperature']}°C
     - Humidity: {weather['humidity']}%
     - Wind Speed: {weather['wind_speed']} m/s
     - Condition: {weather['description']}

     Tree Diagnosis:
     {diagnosis}

     Existing Treatment Plan (already shown to the user separately):
     {plan_text}

     Based on the current weather and diagnosis, provide ONLY these three sections:

     🌧 Disease Risk: Explain the current disease risk in 1-2 short sentences.

     🌦 Weather Impact: Explain how today's weather affects the tree in 1-2 short sentences.

     🕒 Best Treatment Time: Suggest the best time today (or tomorrow if needed) for treatment in 1-2 short sentences.

     Rules:
     - Return ONLY these three sections.
     - Do NOT add recommendations.
     - Do NOT repeat the treatment plan.
     - Do NOT use JSON.
     - Do NOT use Markdown.
     - Maximum 25 words per section.
     - Maximum 75 words total.
     """
     advisory_model = genai.GenerativeModel(
          "gemini-2.5-flash-lite",
          generation_config={"temperature": 0.0}
     )
     try:
          response = advisory_model.generate_content(prompt)
          return response.text
     except Exception as e:
          print(f"Weather advisory generation failed: {e}")
          return None


def generate_weather_advice(weather, diagnosis=None, treatment_plan=None):
     """
     Deterministic, rule-based weather advisory. Costs zero API calls.
     Replaces the previous per-scan Gemini call (see generate_weather_advice_ai
     above) — the three sections only ever depended on humidity/temperature/
     wind thresholds, so there's no need to pay for an LLM call on every scan.

     Returns a dict: {"disease_risk": str, "weather_impact": str, "best_treatment_time": str}
     diagnosis/treatment_plan are accepted for signature compatibility and
     future use, but aren't required for the current rule set.
     """
     humidity = weather.get("humidity", 0)
     temperature = weather.get("temperature", 0)
     wind = weather.get("wind_speed", 0)

     # --- Disease Risk ---
     if humidity >= 85:
          disease_risk = "High fungal disease risk — humidity is very high. Watch closely for new spots or lesions."
     elif humidity >= 70:
          disease_risk = "Moderate fungal disease risk from elevated humidity. Check affected areas regularly."
     else:
          disease_risk = "Low disease risk — current humidity is not favorable for fungal spread."

     # --- Weather Impact ---
     impact_parts = []
     if temperature >= 38:
          impact_parts.append("high heat may add extra stress to the tree")
     elif temperature <= 5:
          impact_parts.append("cold conditions may slow recovery")
     else:
          impact_parts.append("temperature is within a comfortable range")
     if wind >= 10:
          impact_parts.append("strong winds may reduce spray/treatment effectiveness")
     else:
          impact_parts.append("winds are calm enough for treatment")
     weather_impact = "Today, " + " and ".join(impact_parts) + "."

     # --- Best Treatment Time ---
     if wind >= 10 or temperature >= 38:
          best_treatment_time = "Early morning or late evening is best — avoid midday heat and wind."
     elif humidity >= 85:
          best_treatment_time = "Treat as soon as possible; delaying risks further fungal spread."
     else:
          best_treatment_time = "Any time in daylight works well; morning is still ideal."

     return {
          "disease_risk": disease_risk,
          "weather_impact": weather_impact,
          "best_treatment_time": best_treatment_time,
     }