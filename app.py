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

def process_predictions(df, model_id, api):
    """Process predictions for each transaction in the dataframe"""
    results = []
    raw_responses = []  # Store raw responses for debugging
    progress_bar = st.progress(0)
    
    # Create a container for debug output
    if st.session_state.get('debug_mode'):
        debug_container = st.expander("Debug Information (LLM Interactions)", expanded=True)
    
    # Process transactions in batches of 3
    batch_size = 3
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Create transaction details for batch
        batch_details = []
        for _, row in batch.iterrows():
            batch_details.append({
                "Description": row['Description'],
                "Category": row['Category']
            })
        
        # Get predictions for batch
        prediction, error = api.predict(model_id, batch_details)
        raw_responses.extend([prediction] * len(batch))
        
        # Show debug information
        if st.session_state.get('debug_mode'):
            with debug_container:
                st.markdown(f"### Batch {i//batch_size + 1}")
                
                # Show input prompt
                st.markdown("**🔤 Input Prompt:**")
                st.code(api.last_prompt if hasattr(api, 'last_prompt') else "Prompt not available", language="text")
                
                # Show raw output
                st.markdown("**📝 Raw LLM Response:**")
                st.code(prediction if prediction else f"Error: {error}", language="text")
                
                # Show parsed output
                if prediction:
                    st.markdown("**🔍 Parsed Response:**")
                    try:
                        parsed = json.loads(prediction)
                        st.json(parsed)
                    except json.JSONDecodeError:
                        st.error("Failed to parse model output as JSON")
                st.markdown("---")
        
        if error:
            results.extend([{"error": error}] * len(batch))
        else:
            try:
                parsed = json.loads(prediction)
                results.extend(parsed if isinstance(parsed, list) else [parsed])
            except json.JSONDecodeError:
                results.extend([{"error": "Failed to parse prediction"}] * len(batch))
        
        # Update progress
        progress_bar.progress((i + len(batch)) / len(df))
    
    # Add predictions to dataframe
    df['Predicted_Vendor'] = [r.get('Vendor', '') if isinstance(r, dict) else '' for r in results]
    df['Predicted_TaxCategory'] = [r.get('TaxCategory', '') if isinstance(r, dict) else '' for r in results]
    
    # Store predictions in database
    db = Database()
    db.store_predictions(df, results, model_id, raw_responses)
    
    return df

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
                
                if st.button("Get Predictions"):
                    with st.spinner("Processing predictions..."):
                        # Only process first 5 rows
                        df_sample = df_prepared.head()
                        result_df = process_predictions(df_sample, selected_model, api)
                        
                        # Show results
                        st.subheader("Results")
                        st.dataframe(result_df)
                        
                        # Download button for results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error preparing data: {str(e)}")
    else:
        st.warning("No models available")

def get_api_key(provider: ModelProvider) -> str:
    """Get API key for selected provider from .env file"""
    # Map provider to environment variable name
    key_map = {
        ModelProvider.TOGETHER: 'TOGETHER_API_KEY',
        ModelProvider.PREDIBASE: 'PREDIBASE_API_KEY'
    }
    
    env_var = key_map.get(provider)
    if not env_var:
        return None
        
    # First try environment variable
    api_key = os.getenv(env_var)
    
    if not api_key:
        # Try reading from .env file
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith(f'{env_var}='):
                        api_key = line.split('=')[1].strip()
                        break
    
    return api_key

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
    
    # Get API key for selected provider
    default_key = get_api_key(provider)
    api_key = st.text_input(
        f"{provider.value.title()} API Key", 
        value=default_key if default_key else "",
        type="password"
    )
    
    if debug_mode:
        st.sidebar.write("Debug Information:")
        st.sidebar.write(f"Selected Provider: {provider.value}")
        st.sidebar.write(f"Default API Key found: {bool(default_key)}")
        st.sidebar.write(f"Current API Key length: {len(api_key) if api_key else 0}")
    
    if api_key:
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
                    # Prepare data
                    try:
                        df_prepared = prepare_amex_data(df, account_type)
                        
                        # Show preview of prepared data
                        st.subheader("Data Preview")
                        st.dataframe(df_prepared.head())
                        
                        if st.button("Predict"):
                            with st.spinner("Processing predictions..."):
                                # Process all rows
                                result_df = process_predictions(df_prepared, None, api)
                                
                                # Show results
                                st.subheader("Results")
                                st.dataframe(result_df)
                                
                                # Download button for results
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Results CSV",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                                
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
        
        with tab2:
            show_predictions_history()

if __name__ == "__main__":
    main()
