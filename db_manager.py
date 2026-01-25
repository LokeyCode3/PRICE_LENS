import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import datetime
import json
import uuid

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def init_db():
    """Initialize the database schema."""
    conn = get_db_connection()
    if not conn:
        print("Skipping DB initialization (connection failed).")
        return

    cur = None
    try:
        cur = conn.cursor()
        
        # Create table for analysis results
        create_table_query = """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id UUID PRIMARY KEY,
            from_date DATE NOT NULL,
            to_date DATE NOT NULL,
            records_used INTEGER NOT NULL,
            cost_status TEXT NOT NULL,
            demand_status TEXT NOT NULL,
            inventory_status TEXT NOT NULL,
            competitor_status TEXT NOT NULL,
            evidence_json JSONB,
            confidence_score FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        conn.commit()
        print("Database schema initialized successfully.")
    except Exception as e:
        print(f"Schema initialization error: {e}")
        conn.rollback()
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        conn.close()

def save_analysis_result(start_date, end_date, result_data):
    """
    Save the analysis result to the database.
    
    Args:
        start_date (date): Start date of analysis
        end_date (date): End date of analysis
        result_data (dict): Dictionary containing analysis results
    
    Returns:
        str: UUID of the inserted record, or None if failed
    """
    conn = get_db_connection()
    if not conn:
        return None

    cur = None
    try:
        cur = conn.cursor()
        record_id = str(uuid.uuid4())
        
        insert_query = """
        INSERT INTO analysis_results (
            id, from_date, to_date, records_used, 
            cost_status, demand_status, inventory_status, competitor_status,
            evidence_json, confidence_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Prepare data
        records = result_data.get('records', 0)
        cost = result_data.get('cost_status', 'N/A')
        demand = result_data.get('demand_status', 'N/A')
        inventory = result_data.get('inventory_status', 'N/A')
        competitor = result_data.get('competitor_status', 'N/A')
        
        # Optional fields
        evidence = json.dumps(result_data) # Storing full result as evidence for now
        confidence = 0.95 # Simulated confidence score
        
        cur.execute(insert_query, (
            record_id, start_date, end_date, records,
            cost, demand, inventory, competitor,
            evidence, confidence
        ))
        
        conn.commit()
        print(f"Analysis record saved with ID: {record_id}")
        return record_id
        
    except Exception as e:
        print(f"Error saving analysis record: {e}")
        conn.rollback()
        return None
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        conn.close()

def get_all_analysis_results():
    """
    Fetch all analysis results from the database.
    
    Returns:
        list: List of dictionaries containing analysis results
    """
    conn = get_db_connection()
    if not conn:
        return []

    cur = None
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM analysis_results ORDER BY created_at DESC")
        results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error fetching analysis history: {e}")
        return []
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        conn.close()

if __name__ == "__main__":
    init_db()
