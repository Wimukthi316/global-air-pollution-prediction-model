# Global Air Pollution Prediction Model

This project contains a machine learning model to predict PM2.5 AQI (Air Quality Index) values based on other air pollution indicators and country information.

## 📁 Project Files

- `global_air_pollution_model.ipynb` - Main Jupyter notebook with data analysis and model training
- `global_air_pollution_model.py` - Python script version (if needed)
- `model_inference.py` - Standalone script for using the trained models
- `global air pollution dataset.csv` - Original dataset
- **Model Files (Generated):**
  - `random_forest_air_pollution_model.joblib` - Trained Random Forest model
  - `linear_regression_air_pollution_model.joblib` - Trained Linear Regression model
  - `feature_names.joblib` - Feature names for model input

## 🤖 Model Performance

### Random Forest Model (Recommended)
- **R² Score:** 0.7152 (71.5% variance explained)
- **Mean Absolute Error (MAE):** 16.06
- **Root Mean Squared Error (RMSE):** 30.17
- **Number of Features:** 177
- **Number of Trees:** 100

### Linear Regression Model
- **R² Score:** -64,509,351,397,954,504 (Very poor performance)
- **MAE:** 211,544,538.94
- **RMSE:** 14,358,559,862.15
- ⚠️ **Not recommended for use**

## 📊 Dataset Information

- **Total Records:** 23,035 (after cleaning)
- **Features Used:**
  - CO AQI Value
  - Ozone AQI Value  
  - NO2 AQI Value
  - Country (One-hot encoded - 174 countries)
- **Target Variable:** PM2.5 AQI Value

## 🚀 Quick Start

### Loading and Using the Model

```python
import joblib
import pandas as pd

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_air_pollution_model.joblib')
feature_names = joblib.load('feature_names.joblib')

# Example prediction function
def predict_pm25(co_aqi, ozone_aqi, no2_aqi, country='China'):
    # Create input DataFrame
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0  # Initialize with zeros
    
    # Set numerical features
    input_data.loc[0, 'CO AQI Value'] = co_aqi
    input_data.loc[0, 'Ozone AQI Value'] = ozone_aqi
    input_data.loc[0, 'NO2 AQI Value'] = no2_aqi
    
    # Set country (one-hot encoded)
    country_column = f'Country_{country}'
    if country_column in feature_names:
        input_data.loc[0, country_column] = 1
    
    # Make prediction
    prediction = rf_model.predict(input_data)[0]
    return prediction

# Example usage
pm25_prediction = predict_pm25(co_aqi=3, ozone_aqi=45, no2_aqi=8, country='China')
print(f"Predicted PM2.5 AQI: {pm25_prediction:.1f}")
```

### Using the Inference Script

```bash
python model_inference.py
```

## 📋 Data Preprocessing Steps

1. **Data Cleaning:** Removed 428 rows with missing Country/City values (1.8% of data)
2. **Feature Selection:** Removed data leakage columns (AQI Value, AQI Categories)
3. **Dimensionality Reduction:** Dropped City column to avoid curse of dimensionality
4. **Encoding:** Applied One-Hot Encoding to Country column (174 countries)
5. **Train/Test Split:** 80/20 split with random_state=42

## 🎯 Model Features

The model uses 177 features total:
- 3 numerical features (CO AQI, Ozone AQI, NO2 AQI)
- 174 one-hot encoded country features

## 📈 Model Interpretability

The Random Forest model provides feature importance scores. The most important features for predicting PM2.5 AQI are typically:
1. Ozone AQI Value
2. NO2 AQI Value  
3. CO AQI Value
4. Specific country indicators

## ⚙️ Requirements

```
pandas
numpy
scikit-learn
joblib
matplotlib (for visualization)
seaborn (for visualization)
```

## 🔧 Installation

1. Clone or download this repository
2. Install required packages: `pip install pandas numpy scikit-learn joblib matplotlib seaborn`
3. Run the notebook or inference script

## 📝 Notes

- The Random Forest model significantly outperforms Linear Regression due to the complex, non-linear relationships in air pollution data
- Country information is crucial for predictions due to different pollution patterns globally
- Model is trained on historical data and should be updated regularly for best performance
- The model predicts PM2.5 AQI values, which typically range from 0-500+ (higher = worse air quality)

## 🎉 Success!

The models have been successfully trained and saved as joblib files, ready for production deployment or further analysis!