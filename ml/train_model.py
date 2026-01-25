import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# Create directory if it doesn't exist
if not os.path.exists('ml_models'):
    os.makedirs('ml_models')

print("Generating synthetic training data...")
np.random.seed(42)
n_samples = 1000

# Features
raw_material_cost = np.random.normal(400, 50, n_samples)
demand_index = np.random.normal(100, 10, n_samples)
inventory_level = np.random.normal(5000, 1000, n_samples)
competitor_price_avg = np.random.normal(1050, 100, n_samples)

X = pd.DataFrame({
    'raw_material_cost': raw_material_cost,
    'demand_index': demand_index,
    'inventory_level': inventory_level,
    'competitor_price_avg': competitor_price_avg
})

# Target Variable (Price)
# Formula: Cost * 2 + (Demand/100 * 50) + (10000/Inventory) * 10 + (Competitor * 0.1) + Noise
y = (raw_material_cost * 2.0) + \
    ((demand_index / 100.0) * 50.0) + \
    ((10000.0 / inventory_level) * 10.0) + \
    (competitor_price_avg * 0.1) + \
    np.random.normal(0, 5, n_samples) # Noise

print("Training XGBoost Regressor (User Configured)...")
# User requested specific hyperparameters
model = xgb.XGBRegressor(
    n_estimators=50, 
    max_depth=3, 
    learning_rate=0.1,
    objective='reg:squarederror'
)
model.fit(X, y)

print("Saving model to ml_models/pricing_model.json...")
model.save_model("ml_models/pricing_model.json")

# Also save feature names for later use
with open("ml_models/feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("✅ Model training complete.")
print(f"   Model saved: ml_models/pricing_model.json")
