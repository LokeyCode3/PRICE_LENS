import pandas as pd
import xgboost as xgb
import shap
import json
import datetime
import pickle
import time
from pytrends.request import TrendReq
import random

def fetch_real_demand_index(keyword="smartphone"):
    print(f"   Fetching real-time Google Trends data for '{keyword}'...")
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='now 7-d')
        data = pytrends.interest_over_time()
        if not data.empty:
            # Get the most recent value
            latest_value = data[keyword].iloc[-1]
            print(f"   ✅ Real-time Demand Index: {latest_value}")
            return float(latest_value)
    except Exception as e:
        print(f"   ⚠️ API Error (using fallback): {e}")
    
    return 85.0 # Fallback

def main():
    print("===================================================")
    print("STEP 3: LIVE DATA INGESTION & ML PREDICTION")
    print("===================================================")

    # 1. Load Model
    print("1. Loading XGBoost Model...")
    model = xgb.XGBRegressor()
    model.load_model("ml_models/pricing_model.json")
    
    with open("ml_models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # 2. Ingest Live/Real Data
    print("2. Ingesting Data Streams...")
    
    # A. Real Data (Google Trends)
    demand_index = fetch_real_demand_index("smartphone")
    
    # B. Simulated Live Streams (APIs)
    # Simulating a +12% hike in raw materials
    raw_material_cost = 448.0 # Base was 400
    inventory_level = 4200    # Base was 5000
    competitor_price = 1080.0 # Base was 1050
    
    print(f"   Raw Material Cost: ${raw_material_cost} (+12.0%)")
    print(f"   Inventory Level: {inventory_level} units")
    print(f"   Competitor Price: ${competitor_price}")

    # 3. Predict
    input_data = pd.DataFrame([[
        raw_material_cost, 
        demand_index, 
        inventory_level, 
        competitor_price
    ]], columns=feature_names)
    
    predicted_price = model.predict(input_data)[0]
    print(f"\n3. ML Model Prediction: ${predicted_price:.2f}")

    # 4. Explain (SHAP)
    print("4. Computing SHAP Explanations...")
    # We need a background dataset for TreeExplainer, usually. 
    # For TreeExplainer with XGBoost, we can just pass the model.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # SHAP returns matrix, get first row
    sv = shap_values[0]
    
    print("   SHAP Values calculated.")
    
    # 5. Generate Evidence JSON
    print("5. Generating Frozen Evidence Object...")
    
    # Convert SHAP to features_used format
    features_used = []
    total_attribution = 0
    
    for i, name in enumerate(feature_names):
        attribution_val = float(sv[i])
        features_used.append({
            "name": name,
            "value": float(input_data.iloc[0, i]),
            "attribution": attribution_val,
            "data_source": "live_stream_v1"
        })
    
    evidence = {
        "event_id": "live_event_001",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_version": "xgboost_v1",
        "predicted_price": float(predicted_price),
        "inputs": {
            "raw_material_cost": raw_material_cost,
            "demand_index": demand_index,
            "inventory_level": inventory_level,
            "competitor_price_avg": competitor_price
        },
        "features_used": features_used,
        "xai_method": "SHAP"
    }
    
    # Save
    if not os.path.exists('evidence'):
        os.makedirs('evidence')
        
    with open('evidence/evidence_real_event.json', 'w') as f:
        json.dump(evidence, f, indent=2)
        
    print(f"   ✅ Evidence saved to: evidence/evidence_real_event.json")

if __name__ == "__main__":
    import os
    main()
