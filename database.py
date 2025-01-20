import sqlite3
from datetime import datetime
import json

class Database:
    def __init__(self, db_path="transactions.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    category TEXT,
                    payment_account TEXT,
                    predicted_vendor TEXT,
                    predicted_tax_category TEXT,
                    processed_at TIMESTAMP,
                    raw_response TEXT,
                    model_id TEXT
                )
            """)
            conn.commit()
    
    def store_predictions(self, transactions_df, predictions, model_id, raw_responses=None):
        """Store predictions in database"""
        with sqlite3.connect(self.db_path) as conn:
            for idx, row in transactions_df.iterrows():
                prediction = predictions[idx] if idx < len(predictions) else {}
                raw_response = raw_responses[idx] if raw_responses and idx < len(raw_responses) else None
                
                conn.execute("""
                    INSERT INTO transactions (
                        description,
                        category,
                        payment_account,
                        predicted_vendor,
                        predicted_tax_category,
                        processed_at,
                        raw_response,
                        model_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['Description'],
                    row['Category'],
                    row['PaymentAccount'],
                    prediction.get('Vendor', ''),
                    prediction.get('TaxCategory', ''),
                    datetime.now().isoformat(),
                    raw_response,
                    model_id
                ))
            conn.commit()
    
    def get_predictions(self, limit=100):
        """Get recent predictions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    description,
                    category,
                    payment_account,
                    predicted_vendor,
                    predicted_tax_category,
                    processed_at as created_at,
                    model_id
                FROM transactions 
                ORDER BY processed_at DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()] 