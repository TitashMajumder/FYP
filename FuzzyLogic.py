# File: FuzzyLogic.py (ADVANCED OVERHAUL)
import numpy as np # type: ignore
import skfuzzy as fuzz # type: ignore
from skfuzzy import control as ctrl # type: ignore

# --- 1. Define Antecedents (Inputs) ---
# Input 1: The calculated hybrid confidence score (0-100)
confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')

# Input 2: Model Consensus (0 = Strong Disagreement, 100 = Strong Agreement)
consensus = ctrl.Antecedent(np.arange(0, 101, 1), 'consensus')

# Input 3 (NEW): Gemini's Initial Health Status Score (10=Healthy, 50=Stressed, 90=Diseased)
# This allows the rules to reference the initial diagnosis
gemini_status = ctrl.Antecedent(np.arange(0, 101, 1), 'gemini_status')

# --- 2. Define Consequent (Output) ---
reliability = ctrl.Consequent(np.arange(0, 101, 1), 'reliability')

# Final Health Status Output
final_health = ctrl.Consequent(np.arange(0, 101, 1), 'final_health')


# --- 3. Membership Functions ---
# Confidence and Consensus are standard.
confidence['low'] = fuzz.trapmf(confidence.universe, [0, 0, 40, 60])
confidence['medium'] = fuzz.trimf(confidence.universe, [40, 65, 90])
confidence['high'] = fuzz.trapmf(confidence.universe, [75, 90, 100, 100])

consensus['low'] = fuzz.trapmf(consensus.universe, [0, 0, 40, 60])
consensus['medium'] = fuzz.trimf(consensus.universe, [40, 65, 90])
consensus['high'] = fuzz.trapmf(consensus.universe, [75, 90, 100, 100])

reliability['low'] = fuzz.trimf(reliability.universe, [0, 25, 50])
reliability['medium'] = fuzz.trimf(reliability.universe, [30, 50, 70])
reliability['high'] = fuzz.trimf(reliability.universe, [50, 75, 100])

# NEW: Membership Functions for Gemini's Initial Status
# These must match the numerical mapping in get_fuzzy_hybrid_analysis (10, 50, 90)
gemini_status['healthy'] = fuzz.trimf(gemini_status.universe, [0, 10, 30])
gemini_status['stressed'] = fuzz.trimf(gemini_status.universe, [30, 50, 70])
gemini_status['diseased'] = fuzz.trimf(gemini_status.universe, [70, 90, 100])

# Health Output Mapping (same as input status regions)
final_health['healthy'] = fuzz.trimf(final_health.universe, [0, 10, 30])
final_health['stressed'] = fuzz.trimf(final_health.universe, [30, 50, 70])
final_health['diseased'] = fuzz.trimf(final_health.universe, [70, 90, 100])


# --- 4. Define the RULES (Advanced Decision Fusion) ---

# --- Rule Set 1: Consensus (Confirm or slight shift) ---

# 1a. Gemini says HEALTHY. We only output 'Diseased' if confidence is low (an outlier).
rule1 = ctrl.Rule(gemini_status['healthy'] & consensus['high'] & confidence['high'], final_health['healthy'])
rule2 = ctrl.Rule(gemini_status['healthy'] & consensus['low'], final_health['healthy']) 
rule3 = ctrl.Rule(gemini_status['healthy'] & confidence['low'], final_health['healthy']) 
# If a healthy plant has HIGH confidence, but LOW consensus (CNN disagreeing), it stays HEALTHY for safety.


# 1b. Gemini says STRESSED. Consensus dictates if it stays 'Stressed' or shifts to 'Healthy'/'Diseased'.
rule4 = ctrl.Rule(gemini_status['stressed'] & consensus['high'] & confidence['high'], final_health['diseased']) # Upgrade to Diseased if confident agreement
rule5 = ctrl.Rule(gemini_status['stressed'] & consensus['medium'], final_health['stressed']) # Maintain Stressed if medium consensus
rule6 = ctrl.Rule(gemini_status['stressed'] & consensus['low'], final_health['healthy']) # Downgrade to Healthy if low consensus (safety)

# 1c. Gemini says DISEASED. Consensus dictates if it stays 'Diseased' or shifts to 'Stressed'.
rule7 = ctrl.Rule(gemini_status['diseased'] & consensus['high'], final_health['diseased']) # Maintain Diseased if high consensus
rule8 = ctrl.Rule(gemini_status['diseased'] & consensus['low'], final_health['stressed']) # Downgrade to Stressed if low consensus (less severe)


# --- Rule Set 2: Reliability (Always uses Confidence and Consensus) ---
rule_rel1 = ctrl.Rule(confidence['low'] | consensus['low'], reliability['low'])
rule_rel2 = ctrl.Rule(confidence['medium'] & consensus['medium'], reliability['medium'])
rule_rel3 = ctrl.Rule(confidence['high'] & consensus['high'], reliability['high'])


# --- 5. Create and Simulate the System ---
hybrid_ctrl = ctrl.ControlSystem([
     rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
     rule_rel1, rule_rel2, rule_rel3
])
hybrid_sim = ctrl.ControlSystemSimulation(hybrid_ctrl)


def get_fuzzy_hybrid_analysis(confidence_score, gemini_health_label, cnn_health_label):
     """
     Takes model outputs and uses fuzzy logic to determine BOTH 
     the Final Health Status and Reliability.
     """
     
     # 1. Calculate Consensus Score (0-100)
     gemini_is_problem = gemini_health_label in ["Diseased", "Stressed"]
     cnn_is_problem = cnn_health_label.lower() == "diseased"
     
     if (gemini_is_problem == cnn_is_problem):
          consensus_score = 100.0
     else:
          consensus_score = 0.0

     # 2. Map Gemini Health to a Numerical Score (for decision input)
     if gemini_health_label == "Diseased":
          gemini_status_score = 90
     elif gemini_health_label == "Stressed":
          gemini_status_score = 50
     else:
          gemini_status_score = 10
          
     # We now feed the simulation with the CONFIDENCE, CONSENSUS, AND GEMINI'S INITIAL STATUS
     hybrid_sim.input['confidence'] = confidence_score
     hybrid_sim.input['consensus'] = consensus_score
     hybrid_sim.input['gemini_status'] = gemini_status_score # <--- NEW INPUT
     
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
          # Fallback: Returns Gemini's original label and assigns an error flag
          print(f"Fuzzy logic error in hybrid decision: {e}")
          if confidence_score > 75:
               reliability_label = "High (Fallback)"
          else:
               reliability_label = "Medium (Fallback)"
               
          return gemini_health_label, reliability_label