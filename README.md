# 🪙 Silver Price Prediction - India 🇮🇳

An end-to-end machine learning project to predict silver prices for the **Indian market** with:
- **Prices in INR (₹)**
- **GST calculations (3%)**
- **Per gram, per 10g, and per kg pricing**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Features

✅ **Indian Market Prices** - All prices in INR  
✅ **GST Included** - 3% GST as per Indian tax law  
✅ **Multiple Units** - Per gram, per 10 grams, per kg  
✅ **Live Exchange Rate** - Real-time USD/INR conversion  
✅ **ML Predictions** - Next day price forecast  
✅ **Web Interface** - Beautiful, responsive design  
✅ **REST API** - JSON endpoints for integration  

## 💰 GST Information

Silver in India attracts **3% GST** under the Goods and Services Tax Act.

| Price Type | Description |
|------------|-------------|
| Without GST | Base metal price |
| With GST | Final consumer price (3% added) |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/SilverPricePrediction/pipelines/Training_pipeline.py
```

### 3. Run the Web App

```bash
python app.py
```

Open: **http://localhost:8080**

## 📡 API Endpoints

### Get Prediction (Indian Market)

```http
GET /api/predict
```

**Response:**
```json
{
    "success": true,
    "market": "India",
    "currency": "INR",
    "exchange_rate": 83.45,
    "inr_without_gst": {
        "current_per_10g": 7650,
        "predicted_per_10g": 7720,
        "current_per_kg": 765000,
        "predicted_per_kg": 772000
    },
    "inr_with_gst": {
        "current_per_10g": 7880,
        "predicted_per_10g": 7952,
        "current_per_kg": 788000,
        "predicted_per_kg": 795160
    },
    "gst": {
        "rate_percent": 3,
        "per_10g": 230,
        "per_kg": 23000
    }
}
```

### Get Current Price

```http
GET /api/current-price
```

**Response:**
```json
{
    "success": true,
    "market": "India",
    "without_gst": {
        "per_gram": 765,
        "per_10_grams": 7650,
        "per_kg": 765000
    },
    "with_gst": {
        "per_gram": 788,
        "per_10_grams": 7880,
        "per_kg": 788000
    },
    "gst_rate": "3%"
}
```

## 📁 Project Structure

```
Silver-Price-Prediction-India/
├── src/SilverPricePrediction/
│   ├── components/
│   │   ├── Data_ingestion.py       # Fetch silver data
│   │   ├── Data_transformation.py  # Feature engineering
│   │   ├── Model_trainer.py        # Train models
│   │   └── Model_evaluation.py     # Evaluate performance
│   ├── pipelines/
│   │   ├── Training_pipeline.py    # Training workflow
│   │   └── Prediction_Pipeline.py  # Prediction + INR conversion
│   └── utils/utils.py
├── Artifacts_Silver/               # Trained models
├── templates_silver/               # HTML templates
├── static_silver/                  # CSS files
├── app_silver.py                   # Flask app
└── requirements_silver.txt
```

## 🔄 Price Conversion Logic

```
1. Fetch USD price per troy ounce
2. Get live USD/INR exchange rate  
3. Convert to INR per ounce
4. Convert to INR per gram (1 oz = 31.1035g)
5. Calculate per 10g and per kg prices
6. Add 3% GST for final prices
```

## 🤖 ML Models Used

| Model | R² Score |
|-------|----------|
| Lasso | 0.9836 ✅ Best |
| Linear Regression | 0.9808 |
| Ridge | 0.9783 |
| ElasticNet | 0.9700 |

## ⚠️ Disclaimer

**This project is for educational purposes only.**

Actual silver prices at jewellers may include:
- Making charges (8-20%)
- Wastage charges
- Purity variations (925, 999)
- Local market premiums

**Do not use for actual trading decisions.**

## 📧 Contact

Created with ❤️ for the Indian market

---

*Prices shown are indicative. GST @ 3% as applicable on silver in India.*
