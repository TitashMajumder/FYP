# File: FuzzyLogic.py (ADVANCED & LOGICALLY CORRECTED)
import numpy as np # type: ignore
import skfuzzy as fuzz # type: ignore
from skfuzzy import control as ctrl # type: ignore

# --- 1. Define Antecedents (Inputs) ---
# Input 1: The calculated hybrid confidence score (0-100)
confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')

# Input 2: Model Consensus (0 = Strong Disagreement, 100 = Strong Agreement)
consensus = ctrl.Antecedent(np.arange(0, 101, 1), 'consensus')

# Input 3 (NEW): model's Initial Health Status Score (10=Healthy, 50=Stressed, 90=Diseased)
model_status = ctrl.Antecedent(np.arange(0, 101, 1), 'model_status')

# --- 2. Define Consequent (Output) ---
reliability = ctrl.Consequent(np.arange(0, 101, 1), 'reliability')

# Final Health Status Output
final_health = ctrl.Consequent(np.arange(0, 101, 1), 'final_health')


# --- 3. Membership Functions ---
confidence['low'] = fuzz.trapmf(confidence.universe, [0, 0, 40, 60])
confidence['medium'] = fuzz.trimf(confidence.universe, [40, 65, 90])
confidence['high'] = fuzz.trapmf(confidence.universe, [75, 90, 100, 100])

consensus['low'] = fuzz.trapmf(consensus.universe, [0, 0, 40, 60])
consensus['medium'] = fuzz.trimf(consensus.universe, [40, 65, 90])
consensus['high'] = fuzz.trapmf(consensus.universe, [75, 90, 100, 100])

reliability['low'] = fuzz.trimf(reliability.universe, [0, 25, 50])
reliability['medium'] = fuzz.trimf(reliability.universe, [30, 50, 70])
reliability['high'] = fuzz.trimf(reliability.universe, [50, 75, 100])

# NEW: Membership Functions for model's Initial Status (Used for rule inference)
model_status['healthy'] = fuzz.trimf(model_status.universe, [0, 10, 30])
model_status['stressed'] = fuzz.trimf(model_status.universe, [30, 50, 70])
model_status['diseased'] = fuzz.trimf(model_status.universe, [70, 90, 100])

# Health Output Mapping
final_health['healthy'] = fuzz.trimf(final_health.universe, [0, 10, 30])
final_health['stressed'] = fuzz.trimf(final_health.universe, [30, 50, 70])
final_health['diseased'] = fuzz.trimf(final_health.universe, [70, 90, 100])


# --- 4. Define the RULES (Advanced & Corrected Logic) ---

# --- Rule Set 1: Final Health Status (Remains the same as previous advanced fix) ---
rule1 = ctrl.Rule(model_status['healthy'] & consensus['high'] & confidence['high'], final_health['healthy'])
rule2 = ctrl.Rule(model_status['healthy'] & consensus['low'], final_health['healthy']) 
rule3 = ctrl.Rule(model_status['healthy'] & confidence['low'], final_health['healthy']) 
rule4 = ctrl.Rule(model_status['stressed'] & consensus['high'] & confidence['high'], final_health['diseased']) 
rule5 = ctrl.Rule(model_status['stressed'] & consensus['medium'], final_health['stressed']) 
rule6 = ctrl.Rule(model_status['stressed'] & consensus['low'], final_health['healthy']) 
rule7 = ctrl.Rule(model_status['diseased'] & consensus['high'], final_health['diseased']) 
rule8 = ctrl.Rule(model_status['diseased'] & consensus['low'], final_health['stressed']) 


# --- Rule Set 2: Reliability (OVERHAUL TO FIX LOW RELIABILITY ON HEALTHY) ---

# R9: If the status is HEALTHY (model_status is low) and consensus is high, reliability must be HIGH.
rule9 = ctrl.Rule(model_status['healthy'] & consensus['high'], reliability['high']) 

# R10: If the status is HEALTHY but consensus is low (CNN disagrees), reliability is MEDIUM (cautious).
rule10 = ctrl.Rule(model_status['healthy'] & consensus['low'], reliability['medium']) 

# R11: If the status is STRESSED (intermediate uncertainty), reliability is MEDIUM.
rule11 = ctrl.Rule(model_status['stressed'], reliability['medium'])

# R12: If the status is DISEASED, reliability is HIGH only if confidence is also high.
rule12 = ctrl.Rule(model_status['diseased'] & confidence['high'], reliability['high'])
rule13 = ctrl.Rule(model_status['diseased'] & confidence['low'], reliability['low']) 


# --- 5. Create and Simulate the System ---
hybrid_ctrl = ctrl.ControlSystem([
     rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, # Health rules
     rule9, rule10, rule11, rule12, rule13 # NEW RELIABILITY RULES
])
hybrid_sim = ctrl.ControlSystemSimulation(hybrid_ctrl)


def get_fuzzy_hybrid_analysis(confidence_score, model_health_label, cnn_health_label):
     """
     Takes model outputs and uses fuzzy logic to determine BOTH 
     the Final Health Status and Reliability.
     """
     
     # 1. Calculate Consensus Score (0-100)
     model_is_problem = model_health_label in ["Diseased", "Stressed"]
     cnn_is_problem = cnn_health_label.lower() == "diseased"
     
     if (model_is_problem == cnn_is_problem):
          consensus_score = 100.0
     else:
          consensus_score = 0.0

     # 2. Map model Health to a Numerical Score (for decision input)
     if model_health_label == "Diseased":
          model_status_score = 90
     elif model_health_label == "Stressed":
          model_status_score = 50
     else:
          model_status_score = 10
          
     # We now feed the simulation with the CONFIDENCE, CONSENSUS, AND model'S INITIAL STATUS
     hybrid_sim.input['confidence'] = confidence_score
     hybrid_sim.input['consensus'] = consensus_score
     hybrid_sim.input['model_status'] = model_status_score 
     
     try:
          # Run the calculation
          hybrid_sim.compute()
          
          # Get the final "crisp" output scores
          final_health_score = hybrid_sim.output['final_health']
          reliability_score = hybrid_sim.output['reliability']
          
          # 3. Convert Numerical Outputs back to Labels
          
          # Convert Reliability score (0-100) to High/Medium/Low
          if reliability_score > 70:
               reliability_label = "High"
          elif reliability_score > 40:
               reliability_label = "Medium"
          else:
               reliability_label = "Low"
               
          # Convert Final Health score (0-100) to Healthy/Stressed/Diseased
          if final_health_score > 70:
               final_health_label = "Diseased"
          elif final_health_score > 30:
               final_health_label = "Stressed"
          else:
               final_health_label = "Healthy"

          return final_health_label, reliability_label
          
     except Exception as e:
          # Fallback: Returns model's original label and assigns an error flag
          print(f"Fuzzy logic error in hybrid decision: {e}")
          if confidence_score > 75:
               reliability_label = "High (Fallback)"
          else:
               reliability_label = "Medium (Fallback)"
               
          return model_health_label, reliability_label