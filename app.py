import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .low-risk { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .medium-risk { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .high-risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and scaler
@st.cache_resource
def load_resources():
    if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

model, scaler = load_resources()

# Sidebar / Header
st.title("🎓 Student Burnout Prediction System")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📊 Your Daily Habits")
    
    study_hours = st.slider("Study Hours per Day", 0.0, 15.0, 6.0, 0.5)
    sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0, 0.5)
    screen_time = st.slider("Screen Time per Day (Excl. Study)", 0.0, 12.0, 3.0, 0.5)
    stress_level = st.select_slider("Stress Level (1-10)", options=list(range(1, 11)), value=5)
    assignments = st.number_input("Assignments per Week", 0, 20, 3)
    
    predict_btn = st.button("Predict Burnout Risk")

with col2:
    if predict_btn:
        if model and scaler:
            # Prepare input data
            input_df = pd.DataFrame([[study_hours, sleep_hours, screen_time, stress_level, assignments]], 
                                   columns=['Study Hours per Day', 'Sleep Hours per Day', 'Screen Time per Day', 'Stress Level (1-10)', 'Assignments per Week'])
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Get max probability for the predicted class
            max_prob = np.max(probability) * 100
            
            # Display Result
            risk_class = ""
            if prediction == 'Low': risk_class = "low-risk"
            elif prediction == 'Medium': risk_class = "medium-risk"
            else: risk_class = "high-risk"
            
            st.markdown(f"""
            <div class="prediction-card {risk_class}">
                <h2>{prediction} Risk</h2>
                <p>Burnout Probability: <strong>{max_prob:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Advice based on prediction
            st.subheader("💡 Suggestions for You")
            if prediction == 'Low':
                st.success("Great job! You're maintaining a healthy balance. Keep it up!")
                st.write("- Continue prioritizing your sleep.")
                st.write("- Take regular short breaks during study sessions.")
            elif prediction == 'Medium':
                st.warning("You're at moderate risk. It might be time to adjust your schedule.")
                st.write("- Try to increase your sleep by at least 30-60 minutes.")
                st.write("- Reduce non-essential screen time.")
                st.write("- Practice mindfulness or light exercise to manage stress.")
            else:
                st.error("Caution: High burnout risk detected! Immediate changes recommended.")
                st.write("- **Prioritize Sleep:** Aim for at least 7-8 hours consistently.")
                st.write("- **Reduce Load:** Talk to your teachers about assignment deadlines if possible.")
                st.write("- **Digital Detox:** Minimize screen time for 24 hours.")
                st.write("- **Seek Support:** Talk to a counselor or trusted adult about your stress.")
        else:
            st.error("Model files not found. Please run `train_model.py` first.")
    else:
        st.info("← Adjust the sliders and click 'Predict' to see your burnout risk assessment.")
        
st.markdown("---")
with st.expander("📚 What is Student Burnout?"):
    st.write("""
    **Student burnout** is a state of emotional, physical, and mental exhaustion caused by excessive and prolonged stress. 
    It occurs when you feel overwhelmed, emotionally drained, and unable to meet constant demands. 
    
    **Common Symptoms:**
    - Feeling tired all the time
    - Loss of motivation
    - Reduced academic performance
    - Difficulty concentrating
    - Irritability or feeling isolated
    """)
