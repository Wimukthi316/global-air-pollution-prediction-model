import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Global Air Pollution Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading function
@st.cache_resource
def load_models():
    """Load the trained models and feature names."""
    try:
        with st.spinner('Loading AI models... This may take a moment.'):
            rf_model = joblib.load('models/random_forest_air_pollution_model.joblib')
            lr_model = joblib.load('models/linear_regression_air_pollution_model.joblib')
            feature_names = joblib.load('models/feature_names.joblib')
        st.success('Models loaded successfully!')
        return rf_model, lr_model, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Make sure all .joblib files are in the models/ directory.")
        st.info("Please check the repository and ensure all .joblib files are present.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("This might be due to scikit-learn version compatibility issues.")
        return None, None, None

def get_country_list(feature_names):
    """Extract country names from feature names."""
    countries = []
    for feature in feature_names:
        if feature.startswith('Country_'):
            country = feature.replace('Country_', '')
            countries.append(country)
    countries.sort()
    return countries

def predict_pollution(rf_model, lr_model, feature_names, co_aqi, ozone_aqi, no2_aqi, country):
    """Make predictions using both models."""
    # Create input DataFrame
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0  # Initialize with zeros
    
    # Set numerical features
    input_data.loc[0, 'CO AQI Value'] = co_aqi
    input_data.loc[0, 'Ozone AQI Value'] = ozone_aqi
    input_data.loc[0, 'NO2 AQI Value'] = no2_aqi
    
    # Set country (one-hot encoded)
    if country != "Other/Unknown":
        country_column = f'Country_{country}'
        if country_column in feature_names:
            input_data.loc[0, country_column] = 1
    
    # Make predictions
    rf_prediction = rf_model.predict(input_data)[0]
    lr_prediction = lr_model.predict(input_data)[0]
    
    return rf_prediction, lr_prediction

def get_aqi_category(aqi_value):
    """Convert AQI value to category and color."""
    if aqi_value <= 50:
        return "Good", "#00E400"
    elif aqi_value <= 100:
        return "Moderate", "#FFFF00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi_value <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

def main():
    # Load models
    rf_model, lr_model, feature_names = load_models()
    
    if rf_model is None:
        st.error("⚠️ Failed to load models. Please try refreshing the page.")
        st.info("If the problem persists, the models might be too large for Streamlit Cloud's free tier.")
        st.stop()
    
    # Title and description
    st.title("🌍 Global Air Pollution Prediction Model")
    st.markdown("""
    This application predicts **PM2.5 AQI (Air Quality Index)** values based on other air pollution indicators 
    and country information using machine learning models trained on global air pollution data.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Input Parameters")
    
    # Get country list
    countries = get_country_list(feature_names)
    countries.insert(0, "Other/Unknown")
    
    # Input controls
    country = st.sidebar.selectbox(
        "Select Country:",
        countries,
        index=0,
        help="Select the country for prediction. If your country is not listed, choose 'Other/Unknown'."
    )
    
    st.sidebar.markdown("### Pollution Indicators")
    
    co_aqi = st.sidebar.slider(
        "CO AQI Value:",
        min_value=0,
        max_value=50,
        value=1,
        help="Carbon Monoxide Air Quality Index (typically 0-10 for most areas)"
    )
    
    ozone_aqi = st.sidebar.slider(
        "Ozone AQI Value:",
        min_value=0,
        max_value=200,
        value=35,
        help="Ground-level Ozone Air Quality Index"
    )
    
    no2_aqi = st.sidebar.slider(
        "NO2 AQI Value:",
        min_value=0,
        max_value=100,
        value=3,
        help="Nitrogen Dioxide Air Quality Index"
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Predict PM2.5 AQI", type="primary"):
        # Make predictions
        rf_prediction, lr_prediction = predict_pollution(
            rf_model, lr_model, feature_names, co_aqi, ozone_aqi, no2_aqi, country
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Random Forest Prediction (Recommended)")
            rf_category, rf_color = get_aqi_category(rf_prediction)
            
            # Create gauge chart for Random Forest
            fig_rf = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = rf_prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "PM2.5 AQI Value"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 300]},
                    'bar': {'color': rf_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#E8F5E8"},
                        {'range': [50, 100], 'color': "#FFF2E8"},
                        {'range': [100, 150], 'color': "#FFE8E8"},
                        {'range': [150, 200], 'color': "#FFD6D6"},
                        {'range': [200, 300], 'color': "#E8D6FF"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150
                    }
                }
            ))
            fig_rf.update_layout(height=300)
            st.plotly_chart(fig_rf, use_container_width=True)
            
            st.markdown(f"""
            **Prediction:** {rf_prediction:.1f}  
            **Category:** <span style="color: {rf_color}; font-weight: bold;">{rf_category}</span>  
            **Model Accuracy:** R² = 0.715 (71.5%)
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Linear Regression Prediction")
            lr_category, lr_color = get_aqi_category(abs(lr_prediction))
            
            # Create gauge chart for Linear Regression
            fig_lr = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = abs(lr_prediction) if lr_prediction < 0 else lr_prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "PM2.5 AQI Value"},
                gauge = {
                    'axis': {'range': [None, 300]},
                    'bar': {'color': lr_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#E8F5E8"},
                        {'range': [50, 100], 'color': "#FFF2E8"},
                        {'range': [100, 150], 'color': "#FFE8E8"},
                        {'range': [150, 200], 'color': "#FFD6D6"},
                        {'range': [200, 300], 'color': "#E8D6FF"}
                    ]
                }
            ))
            fig_lr.update_layout(height=300)
            st.plotly_chart(fig_lr, use_container_width=True)
            
            st.markdown(f"""
            **Prediction:** {lr_prediction:.1f}  
            **Category:** <span style="color: {lr_color}; font-weight: bold;">{lr_category}</span>  
            **Model Accuracy:** Poor (Not Recommended)
            """, unsafe_allow_html=True)
            
            if lr_prediction < 0:
                st.warning("⚠️ Linear Regression produced a negative value, which indicates poor model performance.")
        
        # Comparison chart
        st.subheader("📈 Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'Linear Regression'],
            'Prediction': [rf_prediction, max(0, lr_prediction)],
            'R² Score': [0.7152, -64509351397954504.0],
            'MAE': [16.06, 211544538.94]
        })
        
        fig_comparison = px.bar(
            comparison_df, 
            x='Model', 
            y='Prediction',
            title='PM2.5 AQI Predictions Comparison',
            color='Model',
            color_discrete_map={'Random Forest': '#2E8B57', 'Linear Regression': '#CD853F'}
        )
        fig_comparison.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Input summary
        st.subheader("📋 Input Summary")
        input_summary = pd.DataFrame({
            'Parameter': ['Country', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value'],
            'Value': [country, co_aqi, ozone_aqi, no2_aqi]
        })
        st.dataframe(input_summary, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About the Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌟 Random Forest Model (Recommended)
        - **R² Score:** 0.7152 (71.5% variance explained)
        - **Mean Absolute Error:** 16.06
        - **RMSE:** 30.17
        - **Features:** 177 (3 numerical + 174 countries)
        - **Trees:** 100
        """)
    
    with col2:
        st.markdown("""
        ### 📉 Linear Regression Model
        - **R² Score:** Very Poor (Negative)
        - **Mean Absolute Error:** 211,544,538.94
        - **RMSE:** 14,358,559,862.15
        - **Status:** ❌ Not recommended for use
        - **Issue:** Cannot handle complex non-linear relationships
        """)
    
    st.markdown("""
    ### 🎯 AQI Categories
    - **Good (0-50):** 🟢 Air quality is satisfactory
    - **Moderate (51-100):** 🟡 Air quality is acceptable for most people
    - **Unhealthy for Sensitive Groups (101-150):** 🟠 Sensitive individuals may experience problems
    - **Unhealthy (151-200):** 🔴 Everyone may begin to experience health effects
    - **Very Unhealthy (201-300):** 🟣 Health alert: everyone may experience serious health effects
    - **Hazardous (301+):** 🔴 Health emergency: everyone may experience serious health effects
    
    ### 📊 Dataset Information
    - **Total Records:** 23,035 air quality measurements
    - **Countries:** 174 countries worldwide
    - **Time Period:** Global air pollution data
    - **Target:** PM2.5 AQI Value prediction
    """)

if __name__ == "__main__":
    main()