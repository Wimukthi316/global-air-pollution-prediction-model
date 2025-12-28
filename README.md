# 🌍 Global Air Pollution Prediction Model

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An AI-powered web application that predicts PM2.5 Air Quality Index using machine learning**

[🚀 **Try Live Demo**](https://global-air-pollution-prediction-modelgit-g5cgdke6fimaqc9anmhqn.streamlit.app/) | [📊 View Notebook](notebooks/global_air_pollution_model.ipynb) | [📈 Model Performance](#-model-performance)

</div>

---

## ✨ Features

- 🤖 **AI-Powered Predictions**: Uses Random Forest algorithm with 71.5% accuracy
- 🌐 **Global Coverage**: Supports 174+ countries worldwide  
- 📱 **Interactive Web App**: Beautiful Streamlit interface with real-time predictions
- 📊 **Visual Analytics**: Gauge charts and color-coded AQI categories
- 🔄 **Model Comparison**: Compare Random Forest vs Linear Regression performance
- ⚡ **Instant Results**: Get predictions in milliseconds

## 🎯 Quick Demo

<div align="center">

### [🌐 **Live Web Application**](https://global-air-pollution-prediction-modelgit-g5cgdke6fimaqc9anmhqn.streamlit.app/)

*Try the interactive demo! Select a country, adjust pollution indicators, and get instant PM2.5 AQI predictions.*

</div>

## 📁 Project Structure

```
📦 global-air-pollution-prediction-model/
├── 🚀 streamlit_app.py              # Interactive web application
├── 📋 requirements.txt              # Python dependencies
├── 📂 models/                       # Trained ML models
│   ├── 🤖 random_forest_air_pollution_model.joblib
│   ├── 📈 linear_regression_air_pollution_model.joblib
│   └── 🏷️ feature_names.joblib
├── 📂 notebooks/                    # Jupyter notebooks
│   └── 📓 global_air_pollution_model.ipynb
├── 📂 data/                         # Dataset
│   └── 📊 global air pollution dataset.csv
├── 📂 src/                          # Source code
│   ├── 🐍 global_air_pollution_model.py
│   └── 🔮 model_inference.py
└── 📖 README.md                     # This file
```

## 🤖 Model Performance

### 🌟 Random Forest (Recommended)
- **R² Score:** `0.7152` (71.5% variance explained)
- **Mean Absolute Error:** `16.06`
- **RMSE:** `30.17`
- **Features:** 177 (3 numerical + 174 countries)
- **Trees:** 100

### 📉 Linear Regression (Comparison)
- **R² Score:** `Poor` (Negative value)
- **MAE:** `211,544,538.94`
- **Status:** ❌ Not recommended

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 23,035 |
| **Countries** | 174 |  
| **Features** | 177 |
| **Target** | PM2.5 AQI Value |
| **Missing Data** | Cleaned (1.8% removed) |

### Input Features:
- 🏭 **CO AQI Value** - Carbon Monoxide levels
- 🌫️ **Ozone AQI Value** - Ground-level ozone
- 🚗 **NO2 AQI Value** - Nitrogen dioxide levels
- 🌍 **Country** - Geographic location (174 countries)

## 🚀 Quick Start

### 1. Web Application (Easiest)
Visit the [**Live Demo**](https://global-air-pollution-prediction-modelgit-g5cgdke6fimaqc9anmhqn.streamlit.app/) for instant predictions!

### 2. Local Installation
```bash
# Clone repository
git clone https://github.com/Wimukthi316/global-air-pollution-prediction-model.git
cd global-air-pollution-prediction-model

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

### 3. Python API Usage
```python
import joblib
import pandas as pd

# Load models
rf_model = joblib.load('models/random_forest_air_pollution_model.joblib')
feature_names = joblib.load('models/feature_names.joblib')

# Make prediction
def predict_air_quality(co_aqi=1, ozone_aqi=35, no2_aqi=3, country='USA'):
    input_data = pd.DataFrame(columns=feature_names).fillna(0)
    input_data.loc[0, 'CO AQI Value'] = co_aqi
    input_data.loc[0, 'Ozone AQI Value'] = ozone_aqi  
    input_data.loc[0, 'NO2 AQI Value'] = no2_aqi
    
    if f'Country_{country}' in feature_names:
        input_data.loc[0, f'Country_{country}'] = 1
    
    return rf_model.predict(input_data)[0]

# Example
pm25_prediction = predict_air_quality(co_aqi=2, ozone_aqi=45, no2_aqi=5, country='China')
print(f"Predicted PM2.5 AQI: {pm25_prediction:.1f}")
```

## 🎨 AQI Categories

| Range | Category | Color | Health Impact |
|-------|----------|-------|---------------|
| 0-50 | Good | 🟢 | Satisfactory |
| 51-100 | Moderate | 🟡 | Acceptable |
| 101-150 | Unhealthy for Sensitive | 🟠 | Sensitive groups affected |
| 151-200 | Unhealthy | 🔴 | Everyone affected |
| 201-300 | Very Unhealthy | 🟣 | Health alert |
| 301+ | Hazardous | 🔴 | Emergency conditions |

## 🛠️ Technical Details

### Data Processing Pipeline:
1. **Data Cleaning** → Remove 1.8% missing values
2. **Feature Engineering** → Drop data leakage columns  
3. **Encoding** → One-hot encode 174 countries
4. **Model Training** → Random Forest with 100 trees
5. **Validation** → 80/20 train-test split

### Technologies Used:
- **Machine Learning:** scikit-learn, pandas, numpy
- **Web App:** Streamlit, plotly
- **Deployment:** Streamlit Cloud
- **Version Control:** Git, GitHub

## 📈 Performance Metrics

<div align="center">

| Model | R² Score | MAE | RMSE | Recommendation |
|-------|----------|-----|------|----------------|
| **Random Forest** | 0.7152 | 16.06 | 30.17 | ✅ **Recommended** |
| Linear Regression | Poor | 211M+ | 14B+ | ❌ Not suitable |

</div>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Global air pollution dataset contributors
- Streamlit team for the amazing framework
- scikit-learn community for ML tools

## 📞 Contact

**Wimukthi316** - [GitHub Profile](https://github.com/Wimukthi316)

⭐ **Star this repository if it helped you!**

---

<div align="center">

**[🚀 Try the Live Demo](https://global-air-pollution-prediction-modelgit-g5cgdke6fimaqc9anmhqn.streamlit.app/) | [📊 Explore the Code](https://github.com/Wimukthi316/global-air-pollution-prediction-model)**

*Made with ❤️ and 🤖 AI*

*Thank you*

</div>