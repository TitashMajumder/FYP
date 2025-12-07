# File: AIModel.py
import google.generativeai as genai
import os
from PIL import Image
import json
import re
from dotenv import load_dotenv
from utils import sectionize_image  # (Or crop_and_encode, whatever you named it)
# --- 1. CONFIGURE THE GEMINI API KEY ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
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
    Analyzes images, asks Gemini for coordinates, crops the diseased part,
    and returns the results with the cropped image string.
    """
    try:
        image_objects = []
        for path in image_paths_list:
            image_objects.append(Image.open(path))

        # --- UPDATED PROMPT: ASKS FOR COORDINATES & IMAGE INDEX ---
        prompt = (
            "You are a plant disease expert. Analyze these images. "
            "The images may contain one or MORE different plants/trees. "
            "First, provide a brief overall summary. "
            "Then, provide a JSON LIST. "
            "For EACH plant, identify: 'tree_name', 'health_condition', 'confidence_percent', 'brief_analysis', "
            "and crucially: 'image_index' (which image in the list, 0, 1, etc.) "
            "and 'box_2d' (bounding box of the diseased area as [ymin, xmin, ymax, xmax] on a scale of 0-1000). "
            "\n\n"
            "Example JSON Item:"
            "{"
            "  'tree_name': 'Mango',"
            "  'health_condition': 'Diseased',"
            "  'confidence_percent': 90,"
            "  'brief_analysis': 'Anthracnose spots visible.',"
            "  'image_index': 0,"
            "  'box_2d': [200, 350, 450, 600]" 
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
            raw_results = json.loads(json_string)
            overall_details = full_text.split("```json")[0].strip()
            
            # --- LOOP THROUGH RESULTS TO CROP ---
            for res in raw_results:
                # 1. Get the Image Index (default to 0 if missing)
                idx = res.get('image_index', 0)
                
                # Safety check: ensure index is valid
                if idx < len(image_paths_list):
                    current_img_path = image_paths_list[idx]
                    current_img_obj = image_objects[idx]
                    w_img, h_img = current_img_obj.size
                    
                    # 2. Get Coordinates (ymin, xmin, ymax, xmax) 0-1000 scale
                    box = res.get('box_2d', None)
                    
                    if box and len(box) == 4:
                        ymin, xmin, ymax, xmax = box
                        
                        # 3. Convert 0-1000 scale to Pixels
                        # y = (ymin / 1000) * height
                        pixel_y = int((ymin / 1000) * h_img)
                        pixel_x = int((xmin / 1000) * w_img)
                        pixel_h = int(((ymax - ymin) / 1000) * h_img)
                        pixel_w = int(((xmax - xmin) / 1000) * w_img)
                        
                        # 4. CALL YOUR CROP FUNCTION
                        # This generates the base64 string
                        cropped_str = sectionize_image(current_img_path, pixel_x, pixel_y, pixel_w, pixel_h)
                        
                        # 5. Add to result object
                        res['cropped_section'] = cropped_str
                    else:
                        res['cropped_section'] = None
                
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
          You are a plant care expert. A tree has been analyzed with the following details:
          - Tree Name: {tree_name}
          - Condition: {health_condition}
          - Analysis Details: {analysis_details}

          Based *only* on this information, provide a simple, actionable, step-by-step treatment plan
          for a park manager or gardener. Format the plan with markdown bullet points.
          If the tree is 'Healthy', simply recommend standard care.
          """

          response = model.generate_content(prompt)
          return response.text.strip()

     except Exception as e:
          print(f"❌ Error in get_treatment_plan: {str(e)}")
          return "Error: Could not generate treatment plan."

def get_gps_from_stamp(image_path):
     """
     Uses the Gemini model to perform OCR and extract GPS coordinates
     stamped onto an image.
     """
     try:
          img = Image.open(image_path)
          
          # A very specific prompt just for OCR
          prompt = (
               "Analyze this image for any text stamped on it, like from a 'Geo Tag Camera' app. "
               "I am looking for GPS coordinates, which might look like 'Lat: 12.3456 Lon: -78.9012'. "
               "If you find coordinates, return *only* a JSON object with two keys: 'lat' and 'lon'. "
               "The values should be floating-point numbers. "
               "If you find no coordinates, return 'None'."
          )

          # We want a precise, non-creative answer
          ocr_config = genai.GenerationConfig(temperature=0.0)
          
          # Use a new, simple model instance for this specific task
          ocr_model = genai.GenerativeModel(
               'gemini-2.5-flash',
               generation_config=ocr_config
          )
          
          response = ocr_model.generate_content([prompt, img])
          
          # Clean the text
          text = response.text.strip().replace("```json", "").replace("```", "")
          
          if "none" in text.lower() or "{" not in text:
               return None, None
               
          # Parse the JSON
          result_json = json.loads(text)
          lat = result_json.get("lat")
          lon = result_json.get("lon")
          
          if lat and lon:
               return float(lat), float(lon)
          else:
               return None, None

     except Exception as e:
          print(f"❌ Error in get_gps_from_stamp (OCR): {str(e)}")
          return None, None

# --- 6. TEST BLOCK UPDATED ---
if __name__ == "__main__":
     # --- 1. TEST THE NEW GPS OCR FUNCTION ---
     print(f"--- 1. TESTING GPS-FROM-STAMP (OCR) ---")
     # IMPORTANT: Change this to a test image that has a GPS stamp
     test_stamp_image = r"D:\FYP\your_geotagged_image.jpg" 
     
     if os.path.exists(test_stamp_image):
          print(f"Testing OCR on: {test_stamp_image}...")
          lat, lon = get_gps_from_stamp(test_stamp_image)
          if lat and lon:
               print(f"✅ OCR Success: Lat={lat}, Lon={lon}")
          else:
               print("❌ OCR: No stamped coordinates found on this image.")
     else:
          print(f"Stamp test image not found at: {test_stamp_image}")
          print("Please update 'test_stamp_image' path to test the OCR function.")
     
     print("\n" + "="*30 + "\n")

     # --- 2. TEST THE MULTI-TREE ANALYSIS FUNCTION ---
     print(f"--- 2. TESTING MULTI-TREE ANALYSIS ---")
     test_image_paths = [
          r"D:\FYP\3.jpg", 
          # r"D:\FYP\trunk_image.jpg"
     ]
     
     if all(os.path.exists(p) for p in test_image_paths):
          print(f"Analyzing {len(test_image_paths)} image(s)...")
          
          # --- THIS IS THE FIX ---
          # Get the list and overall details
          results_list, overall_details = analyze_tree_health(test_image_paths)
          
          print(f"\nOverall Details:\n{overall_details}")
          
          if not results_list:
               print("\nNo individual trees identified.")
          else:
               # Loop through each tree found
               for i, res in enumerate(results_list):
                    name = res.get("tree_name", "Unknown")
                    health = res.get("health_condition", "Unknown")
                    confidence = res.get("confidence_percent", 0)
                    details = res.get("brief_analysis", "No details")

                    print(f"\n--- Result {i+1} ---")
                    print(f"Tree Name: {name}")
                    print(f"Health: {health}")
                    print(f"Confidence: {confidence}%")
                    print(f"Details: {details}")
                    
                    # --- 3. TESTING TREATMENT PLAN (inside the loop) ---
                    print(f"\n--- 3. TESTING TREATMENT PLAN (for {name}) ---")
                    if health != "Error":
                         print("Generating treatment plan...")
                         plan = get_treatment_plan(name, health, details)
                         print(f"Recommended Plan:\n{plan}")
                    else:
                         print("Skipping treatment plan due to analysis error.")
     else:
          print(f"One or more analysis test images not found. Please check paths in 'test_image_paths'.")