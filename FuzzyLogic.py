import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 1. Define Variables ---
# We have one input: 'confidence' (from 0 to 100)
# We have one output: 'reliability' (from 0 to 100)
confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
reliability = ctrl.Consequent(np.arange(0, 101, 1), 'reliability')

# --- 2. Define Membership Functions (The "Fuzzy" part) ---
# This describes what "low", "medium", and "high" mean.
# 'low' is a flat line at 100% from 0-40, then fades to 0 by 60.
confidence['low'] = fuzz.trapmf(confidence.universe, [0, 0, 40, 60])
# 'medium' is a triangle that peaks at 65, starts at 40, ends at 90.
confidence['medium'] = fuzz.trimf(confidence.universe, [40, 65, 90])
# 'high' starts to fade in at 75, and is 100% from 90-100.
confidence['high'] = fuzz.trapmf(confidence.universe, [75, 90, 100, 100])

# We can define simpler outputs for reliability
reliability['low'] = fuzz.trimf(reliability.universe, [0, 25, 50])
reliability['medium'] = fuzz.trimf(reliability.universe, [30, 50, 70])
reliability['high'] = fuzz.trimf(reliability.universe, [50, 75, 100])

# --- 3. Define the Rules ---
# "IF confidence is low, THEN reliability is low"
rule1 = ctrl.Rule(confidence['low'], reliability['low'])
# "IF confidence is medium, THEN reliability is medium"
rule2 = ctrl.Rule(confidence['medium'], reliability['medium'])
# "IF confidence is high, THEN reliability is high"
rule3 = ctrl.Rule(confidence['high'], reliability['high'])

# --- 4. Create and Simulate the System ---
reliability_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
reliability_sim = ctrl.ControlSystemSimulation(reliability_ctrl)


def get_fuzzy_reliability_label(confidence_score):
     """
     Takes a numerical confidence score (e.g., 98) and returns
     a human-readable label ("High", "Medium", "Low")
     based on the fuzzy logic rules.
     """
     try:
          # Give the input score to the simulation
          reliability_sim.input['confidence'] = confidence_score
          
          # Run the calculation
          reliability_sim.compute()
          
          # Get the final "crisp" output score (0-100)
          score = reliability_sim.output['reliability']
          
          # Convert the numerical score into a simple label
          if score > 70:
               return "High"
          elif score > 40:
               return "Medium"
          else:
               return "Low"

     except Exception as e:
          print(f"Fuzzy logic error: {e}")
          # Fallback in case of an error
          if confidence_score > 75:
               return "High (Fallback)"
          elif confidence_score > 50:
               return "Medium (Fallback)"
          else:
               return "Low (Fallback)"