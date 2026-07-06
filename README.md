# 🌳 Help the Greens

An AI-powered plant and tree health scanner built with Streamlit. It combines a
custom CNN classifier with a Gemini vision model, fuses their outputs through a
fuzzy logic decision system, and presents the result with GPS tagging, weather
context, disease heatmaps, and exportable reports.

Final Year Project by Titash.

---

## ✨ Features

- **Hybrid diagnosis** — a custom CNN gives a fast Healthy/Diseased signal
  while Gemini identifies species, health condition (Healthy / Stressed /
  Diseased), a bounding box around the affected area, and a treatment plan.
  A fuzzy logic system (`FuzzyLogic.py`) fuses both signals into one final
  verdict plus a reliability score (Low / Medium / High), so the two models
  checking and correcting each other rather than either one being trusted
  blindly.
- **GPS auto-detection** — tries image metadata OCR first, falls back to
  browser geolocation, then to manual lat/lon entry.
- **Weather-aware advisory** — a deterministic, rule-based advisory (no API
  calls) built from live humidity/temperature/wind data, covering disease
  risk, weather impact, and best treatment timing.
- **Grad-CAM heatmaps** — visualizes where the CNN is looking when it flags
  a tree as diseased or stressed.
- **Treatment plans & PDF export** — per-result PDF reports with images,
  diagnosis, and step-by-step treatment, plus a full CSV/PDF export from the
  admin dashboard.
- **Maps** — a standard marker map, a disease-concentration heatmap, and
  locality-level health statistics, all built on the same survey database.
- **Admin dashboard** — scan analytics, critical-alert surfacing for
  high-confidence diseased trees, and database management tools.
- **Self-building training set** — every diagnosed result (cropped disease
  segment or full healthy image) is automatically logged with its label,
  growing a labeled dataset for future model retraining.

---

## 🧱 Tech Stack

| Layer              | Tool                                      |
|--------------------|-------------------------------------------|
| UI                 | Streamlit                                 |
| Custom classifier  | TensorFlow / Keras (binary CNN + Grad-CAM)|
| Vision LLM         | Google Gemini (`gemini-2.5-flash-lite`)   |
| Decision fusion    | scikit-fuzzy (Mamdani fuzzy inference)    |
| Maps               | Folium + streamlit-folium                 |
| PDF generation     | fpdf2                                     |
| Weather data       | OpenWeatherMap API                        |
| Storage            | SQLite (two databases — see below)        |

---

## 📁 Project Structure

```
plant_desease/
├── assets/                                  # Static frontend assets
│   ├── img.py                               # Standalone CLI: batch-rename image files
│   └── styles.css                           # App-wide dark/green theme
├── database/                                # SQLite databases (auto-created)
│   ├── tree_survey.db                       # Scan results (used by dashboard + maps)
│   └── training_dataset.db                  # Auto-collected labeled training images
├── health_classifier/                       # Training data for the CNN (train/val/test splits)
├── keras_tuner_dir/                         # Keras Tuner hyperparameter search artifacts
├── models/                                  # Trained model + supporting metadata
│   ├── plantvillage_tuned_model copy.h5     # CNN model actually loaded by AIModel.py
│   ├── plantvillage_tuned_model.h5          # Backup/alternate copy
│   └── class_labels_combined.json           # CNN class label mapping
├── notebooks/
│   └── plantDiseaseDetection.ipynb          # CNN training pipeline (offline, not used at runtime)
├── segments/                                # Persistent: cropped diagnosis images (training data + history)
├── services/                                # Supporting backend modules
│   ├── LocalityHeatmap.py                   # Locality-wise disease heatmap + stats
│   ├── MapVisualizer.py                     # Standard marker map
│   ├── ReportGenerator.py                   # Survey DB read/write helpers
│   └── WeatherService.py                    # OpenWeatherMap client
├── temp/                                    # Disposable: working copies of uploaded/captured images
├── venv/                                    # Python virtual environment
├── .env                                     # API keys (not committed)
├── AIModel.py                               # CNN inference, Gemini vision calls, Grad-CAM, weather advisory
├── config.py                                # DB paths + schema definitions
├── dashboard.py                             # Main Streamlit app (Scanner + Map + Admin)
├── FuzzyLogic.py                            # Fuzzy logic decision fusion system
├── init_db.py                               # Database initialization
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root with:
   ```
   model_API_KEY=your_gemini_api_key
   OPENWEATHER_API_KEY=your_openweathermap_api_key
   ```

4. **Make sure `models/` contains:**
   - `plantvillage_tuned_model copy.h5` (the CNN model `AIModel.py` loads by default)
   - `class_labels_combined.json`
   - `optimal_threshold.json` *(optional — if missing, the app falls back to a default threshold of 0.5)*

5. **Run the app**
   ```bash
   streamlit run dashboard.py
   ```

---

## 🗄️ Databases

Both are auto-created on first run (`init_db.py` / `config.py`), no manual setup needed.

- **`database/tree_survey.db`** — `survey` table: every scan result (tree name,
  health, confidence, reliability, GPS, image paths). Powers the dashboard
  history, maps, and admin analytics.
- **`database/training_dataset.db`** — `training_data` table: image path +
  label for every diagnosed result, intended for future model retraining.

---

## 🔍 How a Scan Works

1. **CNN pass** — fast binary Healthy/Diseased check with a confidence score.
2. **Gemini pass** — species identification, detailed health condition,
   bounding box on the affected area, and a treatment plan.
3. **Fuzzy fusion** — `FuzzyLogic.py` combines both signals (agreement level,
   conflict, and confidence) into one final health label and a reliability
   rating, so disagreements between the two models are flagged rather than
   silently resolved in favor of one.
4. **Persistence** — the result is saved to `tree_survey.db`, its cropped
   region (or full image, if healthy) is logged to `training_dataset.db`, and
   it's optionally exported as a PDF.

---

## 📌 Notes

- `temp/` is disposable scratch space the app manages itself — safe to clear
  when the app isn't running.
- `segments/` is **not** disposable — it's the growing training dataset and
  historical record; treat it like data, not cache.
- The weather advisory is fully deterministic (humidity/temperature/wind
  thresholds) and makes no API calls per scan.