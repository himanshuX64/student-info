import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_burnout_model(data_path='student_burnout_data.csv'):
    # Load dataset
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_generation.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Define features and target
    X = df.drop(['Burnout Score', 'Burnout Risk'], axis=1)
    y = df['Burnout Risk']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    # Random Forest is robust and works well with relatively small datasets
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Training Complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model and Scaler saved as 'model.pkl' and 'scaler.pkl'.")
    return model, scaler

if __name__ == "__main__":
    train_burnout_model()
