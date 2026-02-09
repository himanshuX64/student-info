# Student Burnout Prediction System 🎓

A beginner-friendly machine learning project to predict student burnout risk using daily lifestyle inputs.

## 🚀 Features
- **Synthetic Data Generation:** Custom script to create a balanced dataset.
- **Machine Learning Model:** Random Forest Classifier trained to categorize burnout risk.
- **Interactive UI:** Polished Streamlit application with sliders, prediction cards, and personalized advice.
- **Visual Feedback:** Color-coded risk levels (Green: Low, Yellow: Medium, Red: High).

## 🛠️ Project Structure
- `data_generation.py`: Script to generate `student_burnout_data.csv`.
- `train_model.py`: Script to train the model and save `model.pkl` and `scaler.pkl`.
- `app.py`: Main Streamlit application file.
- `requirements.txt`: Python dependencies.

## 🏃 How to Run

### 1. Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### 2. Generate Data and Train Model
Run these scripts in order to set up the model:
```bash
python data_generation.py
python train_model.py
```

### 3. Launch the App
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

## 🧠 Model Features
The system takes the following inputs:
1. **Study Hours:** Mental load per day.
2. **Sleep Hours:** Recovery factor.
3. **Screen Time:** Digital fatigue factor.
4. **Stress Level:** Subjective emotional state.
5. **Assignments:** Weekly academic pressure.

## ⚖️ License
MIT
