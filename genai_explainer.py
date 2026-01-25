from logger import audit_logger

class GenAIExplainer:
    """
    Price Change Explainability AI.
    Strictly follows the regulated environment system context and rules.
    """
    
    def validate_evidence(self, evidence):
        """
        Validates the Evidence JSON against strict requirements.
        Returns True if valid, False otherwise.
        """
        required_fields = [
            'old_price', 'new_price', 'currency', 
            'model_version', 'time_window', 
            'features_used', 'confidence_score', 'safety_flags',
            'xai_method'
        ]
        
        for field in required_fields:
            if field not in evidence:
                audit_logger.log_event("GENERATION_REFUSED", {"reason": f"Missing field: {field}"})
                return False
                
        if evidence['confidence_score'] is None:
             audit_logger.log_event("GENERATION_REFUSED", {"reason": "Missing confidence score"})
             return False

        if evidence.get('xai_method') != 'SHAP':
             audit_logger.log_event("GENERATION_REFUSED", {"reason": "Invalid XAI method (must be SHAP)"})
             return False

        return True

    def safety_filter(self, text, flags):
        """
        MANDATORY: Filters output based on safety flags.
        """
        if flags.get("hide_exact_costs", False):
            text = text.replace("₹", "₹~")
            text = text.replace("INR", "INR~") 
        return text

    def get_confidence_label(self, score):
        if score >= 0.80:
            return "High confidence"
        elif score >= 0.50:
            return "Medium confidence"
        else:
            return "Low confidence"

    def format_explanation(self, evidence, audience):
        """
        Generates the strict 5-section output based on audience type.
        """
        # Data Extraction
        old_price = evidence['old_price']
        new_price = evidence['new_price']
        currency = "₹" if evidence['currency'] == "INR" else evidence['currency']
        from_date = evidence['time_window']['from']
        to_date = evidence['time_window']['to']
        model_version = evidence['model_version']
        confidence_score = evidence['confidence_score']
        features = evidence['features_used']
        xai_method = evidence.get('xai_method', 'Unknown')
        
        # Sort features by attribution magnitude
        sorted_features = sorted(features, key=lambda x: abs(x['attribution']), reverse=True)
        
        # 1. Title
        title = "Price Change Explanation"
        if audience == "regulator":
            title += " (Regulatory Audit)"
        else:
            title += " (Customer Summary)"

        # 2. Price Change Summary
        summary_section = f"**Price Change Summary**\n"
        summary_section += f"• Price: {currency}{old_price} → {currency}{new_price}\n"
        summary_section += f"• Time Window: {from_date} to {to_date}\n"
        if audience == "regulator":
            summary_section += f"• ML Model: {model_version}\n"
        
        # 3. Machine Learning–Based Feature Attribution
        attribution_section = f"**Machine Learning–Based Feature Attribution**\n"
        
        # Narrative Intro
        direction_str = "increase" if new_price > old_price else "decrease"
        if audience == "regulator":
            attribution_section += f"The model detected a price {direction_str} driven by the following factors:\n"
        else:
            attribution_section += f"We adjusted the price due to the following main factors:\n"

        for f in sorted_features:
            impact_pct = int(round(f['attribution'] * 100))
            change_pct = f['value_change_pct']
            
            # Name Mapping
            raw_name = f['name']
            friendly_name = raw_name.replace("_", " ").title()
            if raw_name == "raw_material_cost": friendly_name = "Raw Material Costs"
            if raw_name == "demand_index": friendly_name = "Market Demand"
            if raw_name == "inventory_level": friendly_name = "Inventory Availability"
            if raw_name == "competitor_price_avg": friendly_name = "Competitor Pricing"

            if audience == "regulator":
                # Strict technical detail
                attribution_section += f"• {raw_name}: {impact_pct}% attribution (Input change: {change_pct}%)\n"
            else:
                # Customer friendly
                direction = "increased" if change_pct > 0 else "decreased"
                if raw_name == "competitor_price_avg": direction = "fluctuation"
                attribution_section += f"• {friendly_name}: ≈{impact_pct}% impact (Factor {direction})\n"

        # 4. Confidence Score & Methodology
        label = self.get_confidence_label(confidence_score)
        confidence_section = f"**Confidence Score & Methodology**\n"
        confidence_section += f"• Confidence Level: {label} ({confidence_score})\n"
        confidence_section += f"• Methodology: Feature importance calculated using {xai_method}.\n"
        if audience == "regulator":
            confidence_section += f"• Traceability: All values derived from Model {model_version} via {xai_method}.\n"

        # 5. Disclaimer
        disclaimer_section = f"**Disclaimer**\n"
        disclaimer_section += "This explanation is generated automatically based on model inputs. "
        disclaimer_section += "It does not constitute a legal or binding commitment. "
        if audience == "customer":
            disclaimer_section += "Please contact support for detailed inquiries."

        # Assemble
        full_text = f"# {title}\n\n{summary_section}\n{attribution_section}\n{confidence_section}\n{disclaimer_section}"
        
        return full_text

    def generate_explanations(self, evidence):
        """
        Takes the frozen Evidence Object and produces customer/regulator text.
        """
        
        # 1. Validation
        if not self.validate_evidence(evidence):
            return {
                "customer_text": "Explanation unavailable due to data validation failure.",
                "regulator_text": "Explanation unavailable due to data validation failure.",
                "error": "Validation Failed"
            }

        # 2. Generate Audience-Specific Explanations
        raw_customer = self.format_explanation(evidence, audience="customer")
        raw_regulator = self.format_explanation(evidence, audience="regulator")
        
        # 3. Apply Safety Filter (Mandatory)
        safe_customer = self.safety_filter(raw_customer, evidence['safety_flags'])
        safe_regulator = self.safety_filter(raw_regulator, evidence['safety_flags'])
        
        result = {
            "customer_text": safe_customer,
            "regulator_text": safe_regulator,
            "evidence_used": evidence['event_id']
        }
        
        audit_logger.log_event("TEXT_GENERATED", result)
        
        return result
