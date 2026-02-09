import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generates a synthetic dataset for student burnout prediction.
    
    Logic:
    - Study Hours: 0 to 15 hours/day
    - Sleep Hours: 0 to 12 hours/day
    - Screen Time: 0 to 12 hours/day
    - Stress Level: 1 to 10 scale
    - Assignments: 0 to 20 per week
    
    Target Logic (Score calculation):
    High Study, Low Sleep, High Screen Time, High Stress, High Assignments -> High Burnout
    """
    np.random.seed(seed)
    
    # Generate random features
    study_hours = np.random.uniform(2, 14, n_samples)
    sleep_hours = np.random.uniform(3, 10, n_samples)
    screen_time = np.random.uniform(1, 10, n_samples)
    stress_level = np.random.randint(1, 11, n_samples)
    assignments = np.random.randint(0, 16, n_samples)
    
    # Calculate a base score for burnout propensity
    # Higher score = higher risk
    # Normalized weights: 
    # Study (0.25), Sleep (-0.3), Screen (0.15), Stress (0.2), Assignments (0.1)
    # We invert sleep because less sleep increases burnout
    
    score = (
        (study_hours / 14) * 0.25 + 
        (1 - (sleep_hours / 10)) * 0.3 + 
        (screen_time / 10) * 0.15 + 
        (stress_level / 10) * 0.2 + 
        (assignments / 15) * 0.1
    )
    
    # Add some noise to make it realistic
    noise = np.random.normal(0, 0.05, n_samples)
    score = score + noise
    
    # Clip scores to [0, 1] range
    score = np.clip(score, 0, 1)
    
    # Define thresholds for categorical target
    # Low: 0.0 - 0.4
    # Medium: 0.4 - 0.7
    # High: 0.7 - 1.0
    
    risk_level = []
    for s in score:
        if s < 0.4:
            risk_level.append('Low')
        elif s < 0.7:
            risk_level.append('Medium')
        else:
            risk_level.append('High')
            
    df = pd.DataFrame({
        'Study Hours per Day': np.round(study_hours, 1),
        'Sleep Hours per Day': np.round(sleep_hours, 1),
        'Screen Time per Day': np.round(screen_time, 1),
        'Stress Level (1-10)': stress_level,
        'Assignments per Week': assignments,
        'Burnout Score': np.round(score, 2),
        'Burnout Risk': risk_level
    })
    
    save_path = 'student_burnout_data.csv'
    df.to_csv(save_path, index=False)
    print(f"Dataset generated successfully with {n_samples} samples.")
    print(f"Saved to: {os.path.abspath(save_path)}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
