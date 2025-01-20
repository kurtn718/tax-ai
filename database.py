import json
from datetime import datetime
import streamlit as st

class Database:
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
    
    def store_predictions(self, transactions_df, predictions, model_id, raw_responses=None):
        """Store predictions in Supabase"""
        if not self.supabase or not st.session_state.user:
            st.warning("Skipping database storage - no Supabase client or user")
            return
            
        try:
            for idx, row in transactions_df.iterrows():
                prediction = predictions[idx] if idx < len(predictions) else {}
                raw_response = raw_responses[idx] if raw_responses and idx < len(raw_responses) else None
                
                # Debug logging
                if st.session_state.get('debug_mode'):
                    st.write(f"Storing prediction {idx}:")
                    st.write({
                        'user_id': st.session_state.user.id,
                        'description': row['Description'],
                        'prediction': prediction
                    })
                
                try:
                    # Create transaction record
                    data = {
                        'user_id': st.session_state.user.id,
                        'description': row['Description'],
                        'category': row['Category'],
                        'payment_account': row['PaymentAccount'],
                        'predicted_vendor': prediction.get('Vendor', ''),
                        'predicted_tax_category': prediction.get('TaxCategory', ''),
                        'processed_at': datetime.now().isoformat(),
                        'raw_response': json.dumps(raw_response) if raw_response else None,
                        'model_id': model_id
                    }
                    
                    if st.session_state.get('debug_mode'):
                        st.write("Inserting data:", data)
                        
                    response = self.supabase.table('transactions').insert(data).execute()
                    
                    if not response.data:
                        raise Exception("No data returned from insert")
                        
                    if st.session_state.get('debug_mode'):
                        st.write(f"Supabase response: {response.data}")
                        
                except Exception as e:
                    st.error(f"Failed to store prediction {idx}: {str(e)}")
                    if st.session_state.get('debug_mode'):
                        import traceback
                        st.code(traceback.format_exc())
                    
        except Exception as e:
            st.error(f"Error in store_predictions: {str(e)}")
            if st.session_state.get('debug_mode'):
                import traceback
                st.code(traceback.format_exc())
    
    def get_predictions(self, limit=100):
        """Get recent predictions from Supabase"""
        if not self.supabase or not st.session_state.user:
            return []
            
        response = self.supabase.table('transactions')\
            .select('*')\
            .eq('user_id', st.session_state.user.id)\
            .order('processed_at', desc=True)\
            .limit(limit)\
            .execute()
            
        return response.data 