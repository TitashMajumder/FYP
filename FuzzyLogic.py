# File: FuzzyLogic.py
import numpy as np # type: ignore
import skfuzzy as fuzz # type: ignore
from skfuzzy import control as ctrl # type: ignore

# --- 1. Antecedents (Inputs) ---
# The calculated hybrid confidence score (0-100)
confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
# Model Consensus (0 = Strong Disagreement, 100 = Strong Agreement)
consensus = ctrl.Antecedent(np.arange(0, 101, 1), 'consensus')
# Conflict between model and CNN
conflict = ctrl.Antecedent(np.arange(0, 101, 1), 'conflict')
conflict['none']   = fuzz.trapmf(conflict.universe, [0,  0,  20, 35])
conflict['mild']   = fuzz.trimf(conflict.universe,  [25, 50, 70])
conflict['severe'] = fuzz.trapmf(conflict.universe, [60, 80, 100, 100])
model_status = ctrl.Antecedent(np.arange(0, 101, 1), 'model_status')

# --- 2. Consequent (Output) ---
reliability = ctrl.Consequent(np.arange(0, 101, 1), 'reliability')
final_health = ctrl.Consequent(np.arange(0, 101, 1), 'final_health')
reliability.defuzzify_method  = 'bisector'
final_health.defuzzify_method = 'mom'

# --- 3. Membership Functions ---
confidence['very_low'] = fuzz.trapmf(confidence.universe, [0, 0, 15, 30])
confidence['low']      = fuzz.trapmf(confidence.universe, [15, 30, 45, 60])
confidence['medium']   = fuzz.trimf(confidence.universe,  [45, 65, 80])
confidence['high']     = fuzz.trapmf(confidence.universe, [70, 85, 100, 100])

consensus['very_low'] = fuzz.trapmf(consensus.universe, [0, 0, 15, 30])
consensus['low']      = fuzz.trapmf(consensus.universe, [15, 30, 45, 60])
consensus['medium']   = fuzz.trimf(consensus.universe,  [45, 65, 80])
consensus['high']     = fuzz.trapmf(consensus.universe, [70, 85, 100, 100])

reliability['low'] = fuzz.trimf(reliability.universe, [0, 25, 50])
reliability['medium'] = fuzz.trimf(reliability.universe, [30, 50, 70])
reliability['high'] = fuzz.trimf(reliability.universe, [50, 75, 100])

# Membership Functions for model's Initial Status
model_status['healthy'] = fuzz.trimf(model_status.universe, [0, 10, 30])
model_status['stressed'] = fuzz.trimf(model_status.universe, [30, 50, 70])
model_status['diseased'] = fuzz.trimf(model_status.universe, [70, 90, 100])

# Health Output Mapping
final_health['healthy'] = fuzz.trimf(final_health.universe, [0, 10, 30])
final_health['stressed'] = fuzz.trimf(final_health.universe, [30, 50, 70])
final_health['diseased'] = fuzz.trimf(final_health.universe, [70, 90, 100])


# --- 4. RULES ---
# --- Rule Set 1: Final Health Status ---
# Ensure every model_status is covered across the confidence/consensus spectrum
rule1 = ctrl.Rule(model_status['healthy'] & consensus['high'], final_health['healthy'])
rule2 = ctrl.Rule(model_status['healthy'] & consensus['low'], final_health['healthy'])
rule3 = ctrl.Rule(model_status['healthy'] & consensus['medium'], final_health['healthy'])
# Stressed logic: High confidence/consensus pushes it to Diseased, otherwise stays Stressed or Healthy
rule4 = ctrl.Rule(model_status['stressed'] & consensus['high'] & confidence['high'], final_health['diseased'])
rule5 = ctrl.Rule(model_status['stressed'] & (consensus['medium'] | consensus['high']), final_health['stressed'])
rule6 = ctrl.Rule(model_status['stressed'] & consensus['low'], final_health['stressed'])
# Diseased logic: Robustly handles all confidence levels
rule7 = ctrl.Rule(model_status['diseased'] & (consensus['high'] | consensus['medium']), final_health['diseased'])
rule8 = ctrl.Rule(model_status['diseased'] & consensus['low'], final_health['stressed'])

# --- Rule Set 2: Reliability ---
# Healthy Reliability
rule9 = ctrl.Rule(model_status['healthy'] & consensus['high'], reliability['high'])
rule10 = ctrl.Rule(model_status['healthy'] & consensus['low'], reliability['medium'])
rule11 = ctrl.Rule(model_status['healthy'] & consensus['medium'], reliability['medium'])
# Stressed Reliability
rule12 = ctrl.Rule(model_status['stressed'], reliability['medium'])
# Diseased Reliability
rule13 = ctrl.Rule(model_status['diseased'] & confidence['high'], reliability['high'])
rule14 = ctrl.Rule(model_status['diseased'] & confidence['medium'], reliability['medium'])
rule15 = ctrl.Rule(model_status['diseased'] & confidence['low'], reliability['low'])
# Conflict rules — severe disagreement tanks reliability
rule16 = ctrl.Rule(conflict['severe'] & confidence['medium'], reliability['low'])
rule17 = ctrl.Rule(conflict['severe'] & confidence['high'],   reliability['medium'])
rule18 = ctrl.Rule(conflict['none']   & confidence['high'],   reliability['high'])
rule19 = ctrl.Rule(conflict['mild']   & confidence['low'],    reliability['low'])
rule20 = ctrl.Rule(confidence['very_low'], reliability['low'])
rule21 = ctrl.Rule(consensus['very_low'] & model_status['diseased'], final_health['stressed'])

hybrid_ctrl = ctrl.ControlSystem([
     rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11,
     rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21
])
hybrid_sim = ctrl.ControlSystemSimulation(hybrid_ctrl)

def get_fuzzy_hybrid_analysis(confidence_score, model_health_label, cnn_health_label):
     """
     Takes model outputs and uses fuzzy logic to determine BOTH 
     the Final Health Status and Reliability.
     """

     # 1. Calculate Consensus Score (0-100)
     CONSENSUS_MAP = {
          ("Diseased", "diseased"): 95.0,
          ("Diseased", "stressed"): 55.0,
          ("Diseased", "healthy"):  20.0,
          ("Stressed", "stressed"): 85.0,
          ("Stressed", "diseased"): 60.0,
          ("Stressed", "healthy"):  40.0,
          ("Healthy",  "healthy"):  95.0,
          ("Healthy",  "stressed"): 50.0,
          ("Healthy",  "diseased"): 15.0,
     }
     consensus_score = CONSENSUS_MAP.get(
          (model_health_label, cnn_health_label.lower()), 50.0
     )

     # Conflict score: label distance between model and CNN
     label_distance = {"Healthy": 0, "Stressed": 1, "Diseased": 2}
     VALID_LABELS = {"Healthy", "Stressed", "Diseased"}
     cnn_label_clean = cnn_health_label.capitalize()
     if cnn_label_clean not in VALID_LABELS:
          print(f"Warning: Unknown CNN label '{cnn_health_label}', defaulting conflict to mild (50)")
          conflict_score = 50.0
     else:
          conflict_score = abs(
               label_distance.get(model_health_label, 1) -
               label_distance.get(cnn_label_clean, 1)
          ) * 50  # Results in 0, 50, or 100

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
     hybrid_sim.input['conflict']     = conflict_score
     try:
          # Run the calculation
          hybrid_sim.compute()
          # Get the final "crisp" output scores
          final_health_score = hybrid_sim.output['final_health']
          reliability_score = hybrid_sim.output['reliability']
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
          print(f"Fuzzy logic error in hybrid decision: {e}")
          # Validate outputs before using them
          fh = hybrid_sim.output.get('final_health', None)
          rl = hybrid_sim.output.get('reliability', None)
          if fh is not None and 0 <= fh <= 100:
               final_health_label = "Diseased" if fh > 70 else "Stressed" if fh > 30 else "Healthy"
          else:
               final_health_label = model_health_label  # safe fallback
          if rl is not None and 0 <= rl <= 100:
               reliability_label = "High" if rl > 70 else "Medium" if rl > 40 else "Low"
          else:
               reliability_label = "High (Fallback)" if confidence_score > 75 else "Medium (Fallback)"
          return final_health_label, reliability_label