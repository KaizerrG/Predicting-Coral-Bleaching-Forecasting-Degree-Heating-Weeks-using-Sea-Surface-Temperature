
---

# 🌊 Sea Surface Temperature (SST) Prediction & Coral Habitat Analysis

This project focuses on predicting **Sea Surface Temperature (SST)** in the Indian Ocean using multiple time series and deep learning models. The predicted SST is further integrated with **deep-sea coral distribution data** to study potential impacts on coral habitats.

## 📌 Objectives

* Predict SST from satellite observation data.
* Compare different forecasting models: **LSTM, BiLSTM, RNN, ARIMA**.
* Integrate SST predictions with **coral occurrence records** to evaluate habitat risk.

---

## 📂 Dataset

1. **Sea Surface Temperature (SST)**

   * Source: `IndianOcean.csv`
   * Features: `time`, `lat`, `lon`, `sst`
   * Filtered for Indian Ocean region and post-2010 data.

2. **Deep-Sea Coral Occurrence Data**

   * Source: NOAA – `deep_sea_corals.csv`
   * Features: `ScientificName`, `latitude`, `longitude`
   * Filtered to Indian Ocean extent.

---

## 🧠 Models Implemented

### 1. **LSTM (Long Short-Term Memory)**

* Input: Year, Month, Latitude, Longitude (12-month sequences).
* Architecture: 2-layer LSTM + Dropout + Dense.
* Output: Next-month SST prediction.

### 2. **BiLSTM (Bidirectional LSTM)**

* Two stacked BiLSTM layers.
* Captures both forward and backward temporal dependencies.

### 3. **RNN (SimpleRNN)**

* Baseline recurrent model for sequence forecasting.

### 4. **ARIMA**

* Traditional time-series approach (p, d, q grid search).
* Baseline comparison for deep learning models.

---

## 📊 Evaluation Metrics

* **R² Score**
* **RMSE (Root Mean Squared Error)**
* **Training/Validation Loss Plots**
* **Predicted vs Actual SST Curves**

---

## 🛠️ Tech Stack

* **Python 3.9+**
* TensorFlow / Keras
* Statsmodels (ARIMA)
* NumPy, Pandas
* Matplotlib / Seaborn

---

📖 **Read more about this project here:*https://ieeexplore.ieee.org/document/10899654* 

---
