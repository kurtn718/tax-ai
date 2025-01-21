import streamlit as st
import pandas as pd
import os
import tempfile
import time
import json
from convert_to_jsonl import convert_csv_to_jsonl
from model_factory import ModelFactory, ModelProvider
from pathlib import Path
import sys
from database import Database
import unicodedata
from auth import init_auth, login, signup, logout, show_auth_ui


REQUIRED_COLUMNS = ['Description', 'Amount', 'Category', 'PaymentAccount', 'Vendor', 'TaxCategory']

def get_sample_data():
    """Read sample data from file"""
    try:
        with open('sample-business-personal-accts.csv', 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading sample data file: {str(e)}")
        return None

def read_csv_with_encoding(file):
    """Try different encodings to read the CSV file"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            if hasattr(file, 'seek'):
                file.seek(0)  # Reset file pointer for each attempt
            df = pd.read_csv(file, encoding=encoding)
            
            # Normalize Unicode characters in Description column
            if 'Description' in df.columns:
                df['Description'] = df['Description'].apply(lambda x: unicodedata.normalize('NFKD', str(x))
                                                          .encode('ascii', 'ignore')
                                                          .decode('ascii')
                                                          if isinstance(x, str) else x)
            
            return df, None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, str(e)
    
    return None, "Unable to read CSV file. Please check the file format."

def validate_csv(df):
    """Validate that CSV has all required columns"""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

def cleanup_temp_files(files):
    """Clean up temporary files"""
    for file in files:
        try:
            os.remove(file)
        except:
            pass

def process_predictions(df, model_id, api, preview_only=True):
    try:
        results = []
        raw_responses = []
        progress_bar = st.progress(0)
        
        # Only process first 2 rows if preview
        rows_to_process = df.head(2) if preview_only else df
        total_rows = len(rows_to_process)
        
        if st.session_state.get('debug_mode'):
            st.write("Processing rows:", total_rows)
        
        for idx, row in rows_to_process.iterrows():
            # Create transaction details
            transaction = {
                "Description": row['Description'],
                "Category": row['Category']
            }
            
            # Get prediction
            prediction, error = api.predict(model_id, [transaction])
            raw_responses.append(prediction)
            
            # Show debug information
            if st.session_state.get('debug_mode'):
                with st.expander(f"Debug: Transaction {idx + 1}", expanded=False):
                    st.markdown("**🔤 Input:**")
                    st.json(transaction)
                    st.markdown("**📝 Raw Response:**")
                    st.code(prediction if prediction else f"Error: {error}")
            
            if error:
                results.append({"error": error})
                st.error(f"Error processing transaction {idx + 1}: {error}")
            else:
                try:
                    # Parse the prediction string into JSON
                    parsed = json.loads(prediction)
                    # Handle both single object and array responses
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                results.append(item)
                            else:
                                results.append({"error": "Invalid response format"})
                    elif isinstance(parsed, dict):
                        results.append(parsed)
                    else:
                        results.append({"error": "Invalid response format"})
                except (json.JSONDecodeError, TypeError) as e:
                    error_msg = f"Failed to parse prediction response: {str(e)}"
                    results.append({"error": error_msg})
                    st.error(f"Error processing transaction {idx + 1}: {error_msg}")
            
            # Update progress
            progress_bar.progress((idx + 1) / total_rows)
        
        # Add predictions to dataframe
        result_df = rows_to_process.copy()
        result_df['Predicted_Vendor'] = [
            r.get('Vendor', 'ERROR') if isinstance(r, dict) and 'error' not in r 
            else r.get('error', 'ERROR') if isinstance(r, dict) 
            else 'ERROR' 
            for r in results
        ]
        result_df['Predicted_TaxCategory'] = [
            r.get('TaxCategory', '') if isinstance(r, dict) and 'error' not in r 
            else '' 
            for r in results
        ]
        
        # Store predictions in database
        db = Database(st.session_state.supabase)
        db.store_predictions(result_df, results, model_id, raw_responses)
        
        return result_df, df[2:] if preview_only else None
        
    except Exception as e:
        st.error(f"Error in process_predictions: {str(e)}")
        if st.session_state.get('debug_mode'):
            import traceback
            st.code(traceback.format_exc())
        return None, None

def prepare_amex_data(df, account_type):
    """Prepare Amex CSV data for predictions"""
    # Add PaymentAccount column based on selection
    df['PaymentAccount'] = account_type
    
    # Map Amex columns to our required format
    df_prepared = pd.DataFrame({
        'Description': df['Description'],
        'Amount': df['Amount'],
        'Category': df['Category'],
        'PaymentAccount': df['PaymentAccount']
    })
    
    return df_prepared

def show_predictions_section(api):
    """Show predictions section"""
    st.header("Get Predictions")
    
    # Get available models
    models, error = api.get_models()
    if error:
        st.error(f"Error loading models: {error}")
        return
    
    if not models:
        st.info("No fine-tuned models available. Please fine-tune a model first.")
        return
    
    # Show models in a table
    model_data = []
    for model in models:
        model_data.append({
            "ID": model["id"],
            "Name": model["output_name"],
            "Created": model["created_at"],
            "Status": model["status"]
        })
    
    if model_data:
        st.dataframe(
            pd.DataFrame(model_data),
            column_config={
                "ID": "Model ID",
                "Name": "Model Name",
                "Created": "Created At",
                "Status": "Status"
            }
        )
        
        # Model selection
        model_options = [m["ID"] for m in model_data]
        selected_model = st.selectbox(
            "Select Model for Predictions",
            options=model_options,
            format_func=lambda x: next(m["Name"] for m in model_data if m["ID"] == x)
        )
        
        # Account type selection
        account_type = st.radio(
            "Select Account Type",
            options=["Personal", "Business"],
            horizontal=True
        )
        
        # File upload for predictions
        uploaded_file = st.file_uploader("Upload Amex CSV", type=['csv'])
        
        if uploaded_file is not None:
            df, error = read_csv_with_encoding(uploaded_file)
            if error:
                st.error(error)
                return
            
            # Prepare data
            try:
                df_prepared = prepare_amex_data(df, account_type)
                
                # Show preview of prepared data
                st.subheader("Data Preview")
                st.dataframe(df_prepared.head())
                
                if st.button("Process First 2 Transactions"):
                    with st.spinner("Processing initial transactions..."):
                        result_df, remaining_df = process_predictions(df_prepared, selected_model, api, preview_only=True)
                        
                        # Show results
                        st.subheader("Preview Results")
                        st.dataframe(result_df)
                        
                        # Show action buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("Process Remaining Transactions"):
                                if remaining_df is not None and not remaining_df.empty:
                                    with st.spinner("Processing remaining transactions..."):
                                        full_result_df, _ = process_predictions(remaining_df, selected_model, api, preview_only=False)
                                        st.subheader("All Results")
                                        st.dataframe(pd.concat([result_df, full_result_df]))
                                        
                                        # Download button for all results
                                        csv = pd.concat([result_df, full_result_df]).to_csv(index=False)
                                        st.download_button(
                                            label="Download All Results CSV",
                                            data=csv,
                                            file_name="all_predictions.csv",
                                            mime="text/csv"
                                        )
                        
                        with col2:
                            if st.button("Custom Mappings", disabled=True):
                                st.info("Custom mappings feature coming soon!")
                        
                        with col3:
                            if st.button("Custom Tax Categories", disabled=True):
                                st.info("Custom tax categories feature coming soon!")
                        
                        # Download button for preview results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Preview Results CSV",
                            data=csv,
                            file_name="preview_predictions.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
    else:
        st.warning("No models available")

def get_env_value(key: str) -> str:
    """Get value from environment or streamlit secrets"""
    # First try environment variable
    value = os.getenv(key)
    
    if not value:
        # Try getting from streamlit secrets
        try:
            value = st.secrets[key]
        except KeyError:
            pass
    
    return value

def show_active_jobs(api):
    """Show active fine-tuning jobs"""
    st.subheader("Active Jobs")
    
    jobs, error = api.get_active_jobs()
    if error:
        st.error(f"Error loading active jobs: {error}")
        return
        
    if not jobs:
        st.info("No active fine-tuning jobs")
        return
        
    # Show jobs in a table
    job_data = pd.DataFrame(jobs)
    st.dataframe(
        job_data,
        column_config={
            "id": "Job ID",
            "name": "Name",
            "status": "Status",
            "progress": "Progress",
            "created_at": "Created At"
        }
    )

def show_predictions_history():
    """Show prediction history from database"""
    st.subheader("Previous Predictions")
    
    db = Database()
    predictions = db.get_predictions()
    
    if predictions:
        df = pd.DataFrame(predictions)
        st.dataframe(
            df,
            column_config={
                "description": "Description",
                "category": "Category",
                "payment_account": "Account",
                "predicted_vendor": "Predicted Vendor",
                "predicted_tax_category": "Predicted Tax Category",
                "created_at": "Processed At",
                "model_id": "Model ID"
            }
        )
        
        # Download button for history
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download History CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction history available")

def show_card_management():
    """Show card management section"""
    st.subheader("💳 Card Management")
    
    # Get user's cards
    cards = st.session_state.supabase.table('cards')\
        .select('*')\
        .eq('user_id', st.session_state.user.id)\
        .execute()
    
    # Show existing cards
    if cards.data:
        # Add a styled header for the cards list
        st.markdown("""
            <div style="
                background-color: #f0f2f6;
                padding: 10px 15px;
                border-radius: 5px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                font-weight: bold;
            ">
                <div style="width: 40%">Card Name</div>
                <div style="width: 40%">Type</div>
                <div style="width: 20%">Actions</div>
            </div>
        """, unsafe_allow_html=True)
        
        for card in cards.data:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{card['name']}**")
            with col2:
                st.write(card['card_type'])
            with col3:
                if st.button("🗑️", key=f"delete_{card['id']}", help="Delete card"):
                    try:
                        # Delete the card
                        st.session_state.supabase.table('cards')\
                            .delete()\
                            .eq('id', card['id'])\
                            .eq('user_id', st.session_state.user.id)\
                            .execute()
                        st.success(f"Deleted card: {card['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting card: {str(e)}")
    else:
        st.info("No cards added yet. Add your first card below!")
    
    # Add new card
    with st.expander("➕ Add New Card"):
        with st.form("add_card"):
            # Card type selection
            card_type = st.selectbox(
                "Card Type",
                options=["", "AMEX - Business", "AMEX - Personal"],
                key="new_card_type"
            )
            
            # Simple name input
            name = st.text_input(
                "Card Name",
                key="new_card_name",
                help="Must be unique"
            )
            
            if st.form_submit_button("Add Card"):
                if not card_type or not name:
                    st.error("Please fill in all fields")
                else:
                    try:
                        response = st.session_state.supabase.table('cards').insert({
                            'user_id': st.session_state.user.id,
                            'name': name,
                            'card_type': card_type
                        }).execute()
                        st.success(f"Added card: {name}")
                        st.rerun()
                    except Exception as e:
                        if "unique_card_name" in str(e):
                            st.error("A card with this name already exists")
                        else:
                            st.error(f"Error adding card: {str(e)}")

def main():
    # Initialize authentication
    init_auth()
    
    if st.session_state.get('debug_mode'):
        st.sidebar.write("Auth Debug:")
        st.sidebar.write("User:", st.session_state.user)
        st.sidebar.write("Has token:", 'auth_token' in st.session_state)
    
    # Show auth status in sidebar
    if st.session_state.user:
        with st.sidebar:
            st.write(f"👤 Logged in as:")
            st.info(st.session_state.user.email)
            if st.button("🚪 Logout", use_container_width=True):
                logout()
        
        # Main app content
        print(sys.executable)
        # Add debug mode toggle at the top
        debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
        # Store debug mode in session state
        st.session_state['debug_mode'] = debug_mode
        
        st.title("Tax Classification Assistant")
        
        # Get API credentials from secrets (no UI needed)
        api_key = get_env_value('PREDIBASE_API_KEY')
        tenant_id = get_env_value('PREDIBASE_TENANT_ID')
        
        if debug_mode:
            st.sidebar.write("Debug Information:")
            st.sidebar.write(f"API Key found: {bool(api_key)}")
            st.sidebar.write(f"Tenant ID found: {bool(tenant_id)}")
        
        # Set environment variable for tenant ID
        if tenant_id:
            os.environ['PREDIBASE_TENANT_ID'] = tenant_id
        
        # Create API client (always use Predibase)
        if api_key and tenant_id:
            with st.spinner("Initializing API client..."):
                api = ModelFactory.create(ModelProvider.PREDIBASE, api_key, debug=debug_mode)
            
            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs([
                "💳 Card Management",
                "📤 Upload Transactions", 
                "📋 View History"
            ])
            
            with tab1:
                show_card_management()
            
            with tab2:
                # Get user's cards for selection
                cards = st.session_state.supabase.table('cards')\
                    .select('*')\
                    .eq('user_id', st.session_state.user.id)\
                    .execute()
                    
                if not cards.data:
                    st.info("Please add a card in the Card Management section first")
                else:
                    # Card selection
                    selected_card = st.selectbox(
                        "Select Card",
                        options=cards.data,
                        format_func=lambda x: x['name']
                    )
                    
                    # File upload
                    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
                    
                    if uploaded_file is not None:
                        df, error = read_csv_with_encoding(uploaded_file)
                        if error:
                            st.error(error)
                        else:
                            try:
                                df_prepared = prepare_amex_data(df, selected_card['card_type'])
                                
                                # Show preview of prepared data
                                st.subheader("Data Preview")
                                st.dataframe(df_prepared.head())
                                
                                if st.button("Process First 2 Transactions"):
                                    with st.spinner("Processing initial transactions..."):
                                        result_df, remaining_df = process_predictions(df_prepared, None, api, preview_only=True)
                                        
                                        # Show results
                                        st.subheader("Preview Results")
                                        st.dataframe(result_df)
                                        
                                        # Show action buttons
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            if st.button("Process Remaining Transactions"):
                                                if remaining_df is not None and not remaining_df.empty:
                                                    with st.spinner("Processing remaining transactions..."):
                                                        full_result_df, _ = process_predictions(remaining_df, None, api, preview_only=False)
                                                        st.subheader("All Results")
                                                        st.dataframe(pd.concat([result_df, full_result_df]))
                                                        
                                                        # Download button for all results
                                                        csv = pd.concat([result_df, full_result_df]).to_csv(index=False)
                                                        st.download_button(
                                                            label="Download All Results CSV",
                                                            data=csv,
                                                            file_name="all_predictions.csv",
                                                            mime="text/csv"
                                                        )
                                        
                                        with col2:
                                            if st.button("Custom Mappings", disabled=True):
                                                st.info("Custom mappings feature coming soon!")
                                        
                                        with col3:
                                            if st.button("Custom Tax Categories", disabled=True):
                                                st.info("Custom tax categories feature coming soon!")
                                        
                                        # Download button for preview results
                                        csv = result_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Preview Results CSV",
                                            data=csv,
                                            file_name="preview_predictions.csv",
                                            mime="text/csv"
                                        )
                            
                            except Exception as e:
                                st.error(f"Error processing data: {str(e)}")
            
            with tab3:
                show_predictions_history()
    else:
        show_auth_ui()

if __name__ == "__main__":
    main()
