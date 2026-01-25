import time
from pricing_model import PricingModel
from attribution_engine import AttributionEngine
from genai_explainer import GenAIExplainer

def main():
    print("🤖 Starting Price Change Explainability Bot (Architecture v2)...")
    print("---------------------------------------------------------------")
    print("   Architecture Locked: Model -> Attribution -> Evidence -> GenAI")
    print("---------------------------------------------------------------")
    
    # Initialize Components
    model = PricingModel()
    attribution = AttributionEngine()
    explainer = GenAIExplainer()
    
    # Initialize Attribution Engine with the starting state
    attribution.analyze_change(model.get_state())

    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n[Cycle {iteration}] Simulating Market...")
            
            # 1. Pricing Model Updates (Black Box Simulation)
            current_state = model.simulate_market_update()
            
            # 2. Attribution Engine (Math -> Evidence)
            evidence = attribution.analyze_change(current_state)
            
            if evidence:
                print("🚨 EVIDENCE OBJECT PRODUCED")
                # Updated for New Data Contract
                print(f"   Price: {evidence['old_price']} -> {evidence['new_price']} {evidence['currency']}")
                print(f"   Features Used: {[f['name'] for f in evidence['features_used']]}")
                
                # 3. GenAI Explainer (Evidence -> Language)
                outputs = explainer.generate_explanations(evidence)
                
                print("\n   👩‍💼 Customer Version:")
                print(f"   \"{outputs['customer_text']}\"")
                
                print("\n   🧑‍⚖️ Regulator Version:")
                print(f"   \"{outputs['regulator_text']}\"")
                
            else:
                print("   No explainable change detected.")
                
            print("   (Sleeping...)")
            time.sleep(2)
            
            if iteration >= 5: 
                print("\n🛑 Stopping demo.")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped.")

if __name__ == "__main__":
    main()
