import json
from attribution_engine import AttributionEngine
from genai_explainer import GenAIExplainer

def run_test_scenario(name, old_state, new_state, description):
    print(f"\n===================================================")
    print(f"TEST SCENARIO: {name}")
    print(f"DESCRIPTION: {description}")
    print(f"===================================================")

    # Initialize
    attribution = AttributionEngine()
    explainer = GenAIExplainer()

    # 1. Feed Previous State
    print(f"Step 1: Setting Baseline State...")
    attribution.analyze_change(old_state)

    # 2. Analyze Change (Attribution Engine)
    print(f"Step 2: Running Attribution Engine (Math)...")
    evidence = attribution.analyze_change(new_state)

    if not evidence:
        print("RESULT: No explainable change detected (or validation failed in Attribution Engine).")
        return

    print("✅ Evidence Object Created (Frozen):")
    print(json.dumps(evidence, indent=2))

    # 3. Generate Explanation (GenAI)
    print(f"\nStep 3: Generating Explanations (Strict Mode)...")
    outputs = explainer.generate_explanations(evidence)

    print("\n----- CUSTOMER OUTPUT -----")
    print(outputs['customer_text'])
    
    print("\n----- REGULATOR OUTPUT -----")
    print(outputs['regulator_text'])

    if "error" in outputs:
        print(f"\n⚠️ SYSTEM ERROR: {outputs['error']}")

def run_tamper_test():
    print(f"\n===================================================")
    print(f"TEST SCENARIO: Tamper Attempt (Missing XAI Method)")
    print(f"DESCRIPTION: Trying to feed an Evidence Object without 'xai_method' to GenAI.")
    print(f"===================================================")
    
    explainer = GenAIExplainer()
    
    # Malformed Evidence (Missing xai_method)
    bad_evidence = {
        "old_price": 100,
        "new_price": 110,
        "currency": "INR",
        "model_version": "v1",
        "time_window": {"from": "2024-01-01", "to": "2024-01-02"},
        "features_used": [{"name": "cost", "attribution": 1.0, "value_change_pct": 10}],
        "confidence_score": 0.9,
        "safety_flags": {}
        # Missing 'xai_method'
    }
    
    outputs = explainer.generate_explanations(bad_evidence)
    print("\n----- OUTPUT -----")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    # SCENARIO 1: Raw Material Spike
    old_state_1 = {
        "price": 1000,
        "inputs": {"raw_material_cost": 100, "demand": 50}
    }
    new_state_1 = {
        "price": 1200,
        "inputs": {"raw_material_cost": 150, "demand": 50} # +50% Cost
    }
    run_test_scenario(
        "Raw Material Spike", 
        old_state_1, 
        new_state_1, 
        "Price increases due to a 50% jump in raw material costs."
    )

    # SCENARIO 2: Safety Filter Check
    # We want to see if exact costs are hidden
    old_state_2 = {
        "price": 5000,
        "inputs": {"competitor_price_avg": 4000}
    }
    new_state_2 = {
        "price": 5000, # No change, but let's force a change logic for test or just rely on inputs?
                       # Wait, analyze_change requires price diff.
        "price": 5500,
        "inputs": {"competitor_price_avg": 4500}
    }
    run_test_scenario(
        "Competitor Adjustment",
        old_state_2,
        new_state_2,
        "Price adjusts to match competitor. Verifying safety filters on currency."
    )

    # SCENARIO 3: Tamper Test
    run_tamper_test()
