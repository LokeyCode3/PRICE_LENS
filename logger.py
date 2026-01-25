import logging
import json
import os
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file="audit_log.jsonl"):
        self.log_file = log_file
        # Ensure log file exists or create it
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass
        
        self.logger = logging.getLogger("PriceChangeAudit")
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def log_event(self, event_type, details):
        """
        Logs an event with a timestamp and structured details.
        
        Args:
            event_type (str): The type of event (e.g., "PRICE_CHANGE_DETECTED", "EXPLANATION_GENERATED").
            details (dict): Structured data about the event.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
            
        self.logger.info(f"Logged event: {event_type}")

# Singleton instance for easy access
audit_logger = AuditLogger()
