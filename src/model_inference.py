"""
Global Air Pollution Prediction Model - Inference Script
This script demonstrates how to load and use the trained models for predictions.
"""

import joblib
import pandas as pd
import numpy as np

def load_models():
    """Load the saved models and feature names."""
    try:
        lr_model = joblib.load('linear_regression_air_pollution_model.joblib')
        rf_model = joblib.load('random_forest_air_pollution_model.joblib')
        feature_names = joblib.load('feature_names.joblib')
        
        print("✓ Models loaded successfully!")
        print(f"✓ Number of features: {len(feature_names)}")
        return lr_model, rf_model, feature_names
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        return None, None, None

def predict_pollution(rf_model, feature_names, co_aqi=1, ozone_aqi=30, no2_aqi=2, country='USA'):
    """
    Make a prediction using the Random Forest model.
    
    Parameters:
    - rf_model: Loaded Random Forest model
    - feature_names: List of feature names
    - co_aqi: CO AQI Value
    - ozone_aqi: Ozone AQI Value  
    - no2_aqi: NO2 AQI Value
    - country: Country name
    
    Returns:
    - Predicted PM2.5 AQI Value
    """
    
    # Create a DataFrame with the input features
    input_data = pd.DataFrame(columns=feature_names)
    
    # Fill with zeros (for one-hot encoded countries)
    input_data.loc[0] = 0
    
    # Set the numerical features
    if 'CO AQI Value' in feature_names:
        input_data.loc[0, 'CO AQI Value'] = co_aqi
    if 'Ozone AQI Value' in feature_names:
        input_data.loc[0, 'Ozone AQI Value'] = ozone_aqi
    if 'NO2 AQI Value' in feature_names:
        input_data.loc[0, 'NO2 AQI Value'] = no2_aqi
    
    # Set the country (one-hot encoded)
    country_column = f'Country_{country}'
    if country_column in feature_names:
        input_data.loc[0, country_column] = 1
    else:
        print(f"Warning: Country '{country}' not found in training data")
    
    # Make prediction
    prediction = rf_model.predict(input_data)[0]
    return prediction

def main():
    """Main function to demonstrate model usage."""
    print("=== Global Air Pollution Prediction Model ===\n")
    
    # Load models
    lr_model, rf_model, feature_names = load_models()
    
    if rf_model is None:
        print("Failed to load models. Make sure the .joblib files are in the current directory.")
        return
    
    # Example predictions
    print("\n=== Example Predictions ===")
    
    # Example 1: Moderate pollution in USA
    prediction1 = predict_pollution(rf_model, feature_names, 
                                  co_aqi=1, ozone_aqi=35, no2_aqi=3, country='USA')
    print(f"Example 1 - USA (CO:1, Ozone:35, NO2:3): PM2.5 AQI = {prediction1:.1f}")
    
    # Example 2: High pollution in China
    prediction2 = predict_pollution(rf_model, feature_names,
                                  co_aqi=5, ozone_aqi=80, no2_aqi=15, country='China')
    print(f"Example 2 - China (CO:5, Ozone:80, NO2:15): PM2.5 AQI = {prediction2:.1f}")
    
    # Example 3: Low pollution in Canada
    prediction3 = predict_pollution(rf_model, feature_names,
                                  co_aqi=0, ozone_aqi=20, no2_aqi=1, country='Canada')
    print(f"Example 3 - Canada (CO:0, Ozone:20, NO2:1): PM2.5 AQI = {prediction3:.1f}")
    
    print("\n🎉 Model inference completed successfully!")
    
    # Model performance summary
    print(f"\n=== Model Information ===")
    print(f"Random Forest Model:")
    print(f"- R² Score: 0.7152 (71.5% variance explained)")
    print(f"- Mean Absolute Error: 16.06")
    print(f"- Root Mean Squared Error: 30.17")
    print(f"- Number of Features: {len(feature_names)}")
    print(f"- Number of Trees: {rf_model.n_estimators}")

if __name__ == "__main__":
    main()