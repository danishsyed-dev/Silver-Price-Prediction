"""
Silver Price Prediction - Flask Web Application (Indian Market)

A web interface for predicting silver prices in INR with GST calculations.
"""

from flask import Flask, request, render_template, jsonify
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.SilverPricePrediction.pipelines.Prediction_Pipeline import (
    PredictPipeline, 
    SilverDataFetcher,
    IndianMarketConverter,
    make_prediction
)
from src.SilverPricePrediction.logger import logging


app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Initialize Indian market converter
indian_converter = IndianMarketConverter()


@app.route("/")
def home():
    """Render the home page."""
    indian_prices = None
    try:
        print("HOME: Creating SilverDataFetcher...")
        fetcher = SilverDataFetcher()
        print("HOME: Calling get_current_price()...")
        current_price_usd = fetcher.get_current_price()
        print(f"HOME: Got price: {current_price_usd}")
        
        if current_price_usd:
            print("HOME: Converting to Indian market...")
            indian_prices = indian_converter.convert_to_indian_market(current_price_usd)
            print(f"HOME: Conversion successful - Rs.{indian_prices.get('inr_per_10_grams_with_gst', 'N/A')}/10g")
        else:
            print("HOME: current_price_usd is None")
    except Exception as e:
        import traceback
        print(f"HOME ERROR: {str(e)}")
        print(f"HOME TRACEBACK: {traceback.format_exc()}")
        indian_prices = None
    
    return render_template("index.html", indian_prices=indian_prices)


@app.route("/predict", methods=["GET", "POST"])
def predict_price():
    """
    Handle prediction requests with Indian market prices.
    
    GET: Show prediction form
    POST: Make prediction and show result with INR prices and GST
    """
    if request.method == "GET":
        # Fetch current market data
        try:
            fetcher = SilverDataFetcher()
            current_price_usd = fetcher.get_current_price()
            
            if current_price_usd:
                indian_prices = indian_converter.convert_to_indian_market(current_price_usd)
            else:
                indian_prices = None
        except:
            indian_prices = None
        
        return render_template("form.html", indian_prices=indian_prices)
    
    else:
        try:
            # Auto-predict using latest market data
            fetcher = SilverDataFetcher()
            latest_data = fetcher.fetch_latest_data(days=100)
            current_price_usd = fetcher.get_current_price()
            
            pipeline = PredictPipeline()
            predicted_price_usd = pipeline.predict_from_history(latest_data)[0]
            
            # Calculate USD change
            price_change_usd = predicted_price_usd - current_price_usd
            pct_change = (price_change_usd / current_price_usd) * 100
            
            # Convert to Indian market prices
            current_indian = indian_converter.convert_to_indian_market(current_price_usd)
            predicted_indian = indian_converter.convert_to_indian_market(predicted_price_usd)
            
            result = {
                # USD prices
                'current_price_usd': round(current_price_usd, 2),
                'predicted_price_usd': round(predicted_price_usd, 2),
                'price_change_usd': round(price_change_usd, 2),
                'pct_change': round(pct_change, 2),
                'direction': 'up' if price_change_usd > 0 else 'down',
                
                # Exchange rate
                'usd_to_inr_rate': current_indian['usd_to_inr_rate'],
                
                # Current prices in INR (per 1 gram)
                'current_inr_per_1g': current_indian['inr_per_gram'],
                'current_inr_per_1g_gst': current_indian['inr_per_gram_with_gst'],
                
                # Current prices in INR (per 10 grams)
                'current_inr_per_10g': current_indian['inr_per_10_grams'],
                'current_inr_per_10g_gst': current_indian['inr_per_10_grams_with_gst'],
                
                # Current prices in INR (per kg)
                'current_inr_per_kg': current_indian['inr_per_kg'],
                'current_inr_per_kg_gst': current_indian['inr_per_kg_with_gst'],
                
                # Predicted prices in INR (per 1 gram)
                'predicted_inr_per_1g': predicted_indian['inr_per_gram'],
                'predicted_inr_per_1g_gst': predicted_indian['inr_per_gram_with_gst'],
                
                # Predicted prices in INR (per 10 grams)
                'predicted_inr_per_10g': predicted_indian['inr_per_10_grams'],
                'predicted_inr_per_10g_gst': predicted_indian['inr_per_10_grams_with_gst'],
                
                # Predicted prices in INR (per kg)
                'predicted_inr_per_kg': predicted_indian['inr_per_kg'],
                'predicted_inr_per_kg_gst': predicted_indian['inr_per_kg_with_gst'],
                
                # GST info
                'gst_rate': current_indian['gst_rate'],
                'gst_per_10g': current_indian['gst_per_10_grams'],
                
                # Price change in INR (per 10g)
                'price_change_inr_10g': round(predicted_indian['inr_per_10_grams'] - current_indian['inr_per_10_grams'], 2),
                'price_change_inr_10g_gst': round(predicted_indian['inr_per_10_grams_with_gst'] - current_indian['inr_per_10_grams_with_gst'], 2),
                
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
            }
            
            return render_template("result.html", result=result)
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return render_template("error.html", error=str(e))


@app.route("/api/predict", methods=["GET"])
def api_predict():
    """
    API endpoint for predictions with Indian market prices.
    Returns JSON response.
    """
    try:
        fetcher = SilverDataFetcher()
        latest_data = fetcher.fetch_latest_data(days=100)
        current_price_usd = fetcher.get_current_price()
        
        pipeline = PredictPipeline()
        predicted_price_usd = pipeline.predict_from_history(latest_data)[0]
        
        price_change_usd = predicted_price_usd - current_price_usd
        pct_change = (price_change_usd / current_price_usd) * 100
        
        # Convert to Indian market
        current_indian = indian_converter.convert_to_indian_market(current_price_usd)
        predicted_indian = indian_converter.convert_to_indian_market(predicted_price_usd)
        
        return jsonify({
            'success': True,
            'market': 'India',
            'currency': 'INR',
            
            # USD prices
            'usd': {
                'current': round(current_price_usd, 2),
                'predicted': round(predicted_price_usd, 2),
                'change': round(price_change_usd, 2),
                'pct_change': round(pct_change, 2),
            },
            
            # Exchange rate
            'exchange_rate': current_indian['usd_to_inr_rate'],
            
            # INR prices without GST
            'inr_without_gst': {
                'current_per_10g': current_indian['inr_per_10_grams'],
                'predicted_per_10g': predicted_indian['inr_per_10_grams'],
                'current_per_kg': current_indian['inr_per_kg'],
                'predicted_per_kg': predicted_indian['inr_per_kg'],
            },
            
            # INR prices with GST
            'inr_with_gst': {
                'current_per_10g': current_indian['inr_per_10_grams_with_gst'],
                'predicted_per_10g': predicted_indian['inr_per_10_grams_with_gst'],
                'current_per_kg': current_indian['inr_per_kg_with_gst'],
                'predicted_per_kg': predicted_indian['inr_per_kg_with_gst'],
            },
            
            # GST info
            'gst': {
                'rate_percent': current_indian['gst_rate'],
                'per_10g': current_indian['gst_per_10_grams'],
                'per_kg': current_indian['gst_per_kg'],
            },
            
            'direction': 'up' if price_change_usd > 0 else 'down',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/current-price", methods=["GET"])
def api_current_price():
    """Get current silver price in INR."""
    try:
        print("DEBUG: Starting api_current_price")
        fetcher = SilverDataFetcher()
        print("DEBUG: Fetcher created, getting price...")
        current_price_usd = fetcher.get_current_price()
        print(f"DEBUG: Got price: {current_price_usd}")
        
        if current_price_usd:
            print("DEBUG: Converting to Indian market...")
            indian_prices = indian_converter.convert_to_indian_market(current_price_usd)
            print(f"DEBUG: Conversion done: {indian_prices}")
            
            return jsonify({
                'success': True,
                'market': 'India',
                'usd_price': indian_prices['usd_price'],
                'exchange_rate': indian_prices['usd_to_inr_rate'],
                
                'without_gst': {
                    'per_gram': indian_prices['inr_per_gram'],
                    'per_10_grams': indian_prices['inr_per_10_grams'],
                    'per_kg': indian_prices['inr_per_kg'],
                },
                'with_gst': {
                    'per_gram': indian_prices['inr_per_gram_with_gst'],
                    'per_10_grams': indian_prices['inr_per_10_grams_with_gst'],
                    'per_kg': indian_prices['inr_per_kg_with_gst'],
                },
                'gst_rate': f"{indian_prices['gst_rate']}%",
                'timestamp': datetime.now().isoformat()
            })
        else:
            print("DEBUG: current_price_usd is None/False")
            return jsonify({
                'success': False,
                'error': 'Could not fetch price from Yahoo Finance'
            }), 500
        
    except Exception as e:
        import traceback
        print(f"DEBUG ERROR: {str(e)}")
        print(f"DEBUG TRACEBACK: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/api/historical", methods=["GET"])
def api_historical():
    """Get historical silver price data."""
    try:
        days = request.args.get('days', 30, type=int)
        days = min(days, 365)  # Limit to 1 year
        
        fetcher = SilverDataFetcher()
        data = fetcher.fetch_latest_data(days=days)
        
        # Convert to list of dicts
        records = data.to_dict('records')
        for record in records:
            record['Date'] = record['Date'].isoformat() if hasattr(record['Date'], 'isoformat') else str(record['Date'])
        
        return jsonify({
            'success': True,
            'data': records,
            'count': len(records)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))  # HF Spaces uses 7860, Render uses 8080
    
    print("\n" + "="*60)
    print(" Silver Price Prediction - Indian Market")
    print("="*60)
    print(f"Starting server at http://localhost:{port}")
    print("Prices shown in INR with GST (3%)")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)
