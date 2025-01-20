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
    """Process predictions for each transaction in the dataframe"""
    results = []
    raw_responses = []
    progress_bar = st.progress(0)
    
    # Only process first 10 rows if preview
    rows_to_process = df.head(10) if preview_only else df
    total_rows = len(rows_to_process)
    
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
                parsed = json.loads(prediction)
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
            except json.JSONDecodeError:
                error_msg = "Failed to parse prediction response"
                results.append({"error": error_msg})
                st.error(f"Error processing transaction {idx + 1}: {error_msg}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_rows)
    
    # Add predictions to dataframe
    result_df = rows_to_process.copy()
    result_df['Predicted_Vendor'] = [r.get('Vendor', 'ERROR') if isinstance(r, dict) and not isinstance(r, str) and 'error' not in r else r.get('error', 'ERROR') for r in results]
    result_df['Predicted_TaxCategory'] = [r.get('TaxCategory', 'ERROR') if isinstance(r, dict) and not isinstance(r, str) and 'error' not in r else '' for r in results]
    
    # Store predictions in database
    db = Database()
    db.store_predictions(result_df, results, model_id, raw_responses)
    
    return result_df, df[10:] if preview_only else None

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
                
                if st.button("Process First 10 Transactions"):
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

def main():
    print(sys.executable)
    # Add debug mode toggle at the top
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    # Store debug mode in session state
    st.session_state['debug_mode'] = debug_mode
    
    st.title("Tax Classification Assistant")
    
    # Provider selection with Predibase as default
    provider = st.radio(
        "Select Provider",
        options=[ModelProvider.PREDIBASE, ModelProvider.TOGETHER],
        format_func=lambda x: x.value.title(),
        horizontal=True,
        index=0
    )
    
    # Get API key and tenant ID for selected provider
    default_key = get_env_value('PREDIBASE_API_KEY' if provider == ModelProvider.PREDIBASE else 'TOGETHER_API_KEY')
    api_key = st.text_input(
        f"{provider.value.title()} API Key", 
        value=default_key if default_key else "",
        type="password"
    )

    # Add tenant ID input for Predibase
    tenant_id = None
    if provider == ModelProvider.PREDIBASE:
        default_tenant = get_env_value('PREDIBASE_TENANT_ID')
        tenant_id = st.text_input(
            "Predibase Tenant ID",
            value=default_tenant if default_tenant else "",
            type="password",
            help="Enter your Predibase tenant ID"
        )
        if tenant_id:
            os.environ['PREDIBASE_TENANT_ID'] = tenant_id
    
    if debug_mode:
        st.sidebar.write("Debug Information:")
        st.sidebar.write(f"Selected Provider: {provider.value}")
        st.sidebar.write(f"Default API Key found: {bool(default_key)}")
        st.sidebar.write(f"Current API Key length: {len(api_key) if api_key else 0}")
        if provider == ModelProvider.PREDIBASE:
            st.sidebar.write(f"Tenant ID: {tenant_id}")
    
    # Only proceed if we have all required credentials
    if api_key and (provider != ModelProvider.PREDIBASE or tenant_id):
        # Create API client based on selected provider
        with st.spinner("Initializing API client..."):
            api = ModelFactory.create(provider, api_key, debug=debug_mode)
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Process Transactions", "View History"])
        
        with tab1:
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
                else:
                    try:
                        df_prepared = prepare_amex_data(df, account_type)
                        
                        # Show preview of prepared data
                        st.subheader("Data Preview")
                        st.dataframe(df_prepared.head())
                        
                        if st.button("Process First 10 Transactions"):
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
        
        with tab2:
            show_predictions_history()

if __name__ == "__main__":
    main()
