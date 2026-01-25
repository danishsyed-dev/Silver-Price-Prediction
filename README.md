# 🪙 Silver Price Prediction - India 🇮🇳

An end-to-end machine learning project to predict silver prices for the **Indian market**.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render)](https://silver-price-prediction-ghx8.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌐 Live Demo

**🚀 Try it now:** [https://silver-price-prediction-ghx8.onrender.com](https://silver-price-prediction-ghx8.onrender.com)

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| ✅ **Live Prices** | Real-time silver prices from MetalpriceAPI |
| ✅ **Indian Market** | Prices in INR (₹) with import duties |
| ✅ **GST Included** | 3% GST as per Indian tax law |
| ✅ **Multiple Units** | Per gram, per 10 grams, per kg |
| ✅ **ML Predictions** | Next-day price forecast |
| ✅ **Responsive Design** | Beautiful web interface |
| ✅ **REST API** | JSON endpoints for integration |
| ✅ **24-Hour Caching** | Efficient API usage |

---

## 💰 Current Pricing (Example)

| Unit | Price (incl. GST) |
|------|-------------------|
| 1 Gram | ~₹365 |
| 10 Grams | ~₹3,650 |
| 1 Kilogram | ~₹3,65,000 |

*Prices match GoodReturns.in (Hyderabad rates)*

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  MetalpriceAPI  │────▶│   Flask App      │────▶│   ML Model      │
│  (Live Prices)  │     │   (Conversion)   │     │   (Prediction)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
   USD/oz prices         INR conversion          Next-day forecast
                         + GST (3%)
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/danishsyed-dev/Silver-Price-Prediction.git
cd Silver-Price-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional)

```bash
# For MetalpriceAPI (optional - falls back to Yahoo Finance)
export METALPRICEAPI_KEY=your_api_key_here
```

### 4. Run the Web App

```bash
python app.py
```

Open: **http://localhost:8080**

---

## 🌐 Deployment (Render.com)

### Environment Variables Required

| Variable | Description |
|----------|-------------|
| `METALPRICEAPI_KEY` | API key from metalpriceapi.com |

### Build Command
```
pip install -r requirements.txt
```

### Start Command
```
gunicorn app:app
```

---

## 📡 API Endpoints

### Get Prediction

```http
GET /api/predict
```

### Get Current Price

```http
GET /api/current-price
```

### Sample Response

```json
{
    "success": true,
    "market": "India",
    "currency": "INR",
    "with_gst": {
        "per_10_grams": 3650,
        "per_kg": 365000
    },
    "gst_rate": "3%"
}
```

---

## 📁 Project Structure

```
Silver-Price-Prediction/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── Artifacts/                      # ML model files
│   ├── model.pkl                   # Trained model (Lasso)
│   ├── preprocessor.pkl            # Data preprocessor
│   └── raw_data.csv                # Historical data
├── src/SilverPricePrediction/
│   ├── pipelines/
│   │   └── Prediction_Pipeline.py  # Core prediction logic
│   └── components/                 # ML components
├── templates/                      # HTML templates
│   ├── index.html                  # Homepage
│   ├── form.html                   # Prediction form
│   ├── result.html                 # Results page
│   └── about.html                  # Methodology
├── static/
│   ├── style.css                   # Styling
│   └── favicon.png                 # Browser tab icon
└── render.yaml                     # Render deployment config
```

---

## 🔄 Price Conversion Logic

```
Step 1: Fetch USD price per troy ounce (MetalpriceAPI)
Step 2: Get live USD/INR exchange rate
Step 3: Convert to INR per gram (÷ 31.1035)
Step 4: Add Import Duty (+6%)
Step 5: Add Local Premium (+10%)
Step 6: Add GST (+3%)
```

---

## 🤖 ML Model

| Metric | Value |
|--------|-------|
| **Algorithm** | Lasso Regression |
| **R² Score** | 0.9836 |
| **Library** | scikit-learn 1.7.0 |

### Features Used
- Historical closing prices (1, 2, 3, 5, 7 days)
- Moving averages (5, 10, 20 days)
- Technical indicators (RSI, MACD, Bollinger Bands)

---

## 💾 Data Sources

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | MetalpriceAPI | Primary (24-hour cache) |
| 2 | Yahoo Finance | Backup (XAGUSD=X, SI=F) |
| 3 | Local CSV | Fallback |

---

## ⚠️ Disclaimer

**This project is for educational purposes only.**

Actual silver prices at jewellers may include:
- Making charges (8-20%)
- Wastage charges
- Purity variations (925, 999)
- Local market premiums

**Do not use for actual trading decisions.**

---

## 📝 Recent Updates

- ✅ Deployed to Render.com
- ✅ Integrated MetalpriceAPI for accurate Indian prices
- ✅ Added 24-hour price caching
- ✅ Added silver favicon (Ag)
- ✅ Fixed scikit-learn compatibility (v1.7.0)

---

## 📧 Contact

Created by **Danish Syed** 

GitHub: [@danishsyed-dev](https://github.com/danishsyed-dev)

---

*Prices shown are indicative. GST @ 3% as applicable on silver in India.*
