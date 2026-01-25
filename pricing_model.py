import random
import xgboost as xgb
import pandas as pd
import pickle
import os
from logger import audit_logger

class PricingModel:
    """
    Simulates the pricing engine using a Real XGBoost Model.
    """
    def __init__(self, initial_price=None):
        self.model_path = "ml_models/pricing_model.json"
        self.features_path = "ml_models/feature_names.pkl"
        self.loaded = False
        
        # Load Model
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
            with open(self.features_path, "rb") as f:
                self.feature_names = pickle.load(f)
            self.loaded = True
            print("✅ PricingModel: Loaded XGBoost model.")
        else:
            print("⚠️ PricingModel: XGBoost model not found. Using fallback logic.")
            self.feature_names = ["raw_material_cost", "demand_index", "inventory_level", "competitor_price_avg"]
            
        # Initial State
        self.current_inputs = {
            "raw_material_cost": 400.0,
            "demand_index": 100.0,
            "inventory_level": 5000,
            "competitor_price_avg": 1050.0
        }
        
        if self.loaded:
            self.current_price = self._predict_price(self.current_inputs)
        else:
            self.current_price = 1000.0
            
        audit_logger.log_event("MODEL_INITIALIZED", {
            "initial_price": self.current_price,
            "initial_inputs": self.current_inputs,
            "model_type": "XGBoost" if self.loaded else "Fallback"
        })

    def _predict_price(self, inputs):
        if not self.loaded:
            # Fallback (Dummy formula)
            return 1000.0 
            
        # Create DataFrame in correct order
        df = pd.DataFrame([inputs], columns=self.feature_names)
        prediction = self.model.predict(df)[0]
        return float(round(prediction, 2))

    def get_state(self):
        return {
            "price": self.current_price,
            "inputs": self.current_inputs.copy()
        }
        
    def get_model(self):
        """Returns the raw XGBoost model object (for Explainability)"""
        if self.loaded:
            return self.model
        return None

    def simulate_market_update(self):
        """
        Updates inputs and calculates new price using ML model.
        """
        new_inputs = self.current_inputs.copy()
        
        # Randomly fluctuate one or more inputs
        change_type = random.choice(["cost_hike", "demand_surge", "inventory_drop", "competitor_move", "mixed"])
        
        if change_type == "cost_hike":
            new_inputs["raw_material_cost"] *= 1.062 # +6.2%
        elif change_type == "demand_surge":
            new_inputs["demand_index"] *= 1.098 # +9.8%
        elif change_type == "inventory_drop":
            new_inputs["inventory_level"] = int(new_inputs["inventory_level"] * 0.879) # -12.1%
        elif change_type == "competitor_move":
            new_inputs["competitor_price_avg"] *= 1.031 # +3.1%
        elif change_type == "mixed":
            new_inputs["raw_material_cost"] *= 1.02
            new_inputs["competitor_price_avg"] *= 0.98

        # Calculate new price using XGBoost
        new_price = self._predict_price(new_inputs)

        self.current_price = new_price
        self.current_inputs = new_inputs
        
        return self.get_state()
