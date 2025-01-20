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
            # Reset file pointer to start
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
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
    progress_bar = st.progress(0)
    
    # Create a container for debug output
    if st.session_state.get('debug_mode'):
        debug_container = st.expander("Debug Information", expanded=True)
    
    for idx, row in df.iterrows():
        # Create transaction details
        details = {
            "Description": row['Description'],
            "Amount": row['Amount'],
            "Category": row['Category'],
            "PaymentAccount": row['PaymentAccount']
        }
        
        # Get prediction
        prediction, error = api.predict(model_id, details)
        
        if st.session_state.get('debug_mode'):
            with debug_container:
                st.markdown(f"### Transaction {idx + 1}")
                st.markdown("**Input:**")
                st.json(details)
                st.markdown("**Raw Model Output:**")
                st.text(prediction if prediction else error)
                if prediction:
                    st.markdown("**Parsed Response:**")
                    try:
                        parsed = json.loads(prediction)
                        st.json(parsed)
                    except json.JSONDecodeError:
                        st.error("Failed to parse model output as JSON")
                st.markdown("---")
        
        if error:
            results.append({"error": error})
        else:
            try:
                results.append(json.loads(prediction))
            except json.JSONDecodeError:
                results.append({"error": "Failed to parse prediction"})
        
        # Update progress
        progress_bar.progress((idx + 1) / len(df))
    
    # Add predictions to dataframe
    df['Predicted_Vendor'] = [r.get('vendor', '') if isinstance(r, dict) else '' for r in results]
    df['Predicted_TaxCategory'] = [r.get('tax_category', '') if isinstance(r, dict) else '' for r in results]
    
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

def main():
    print(sys.executable)
    # Add debug mode toggle at the top
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    # Store debug mode in session state
    st.session_state['debug_mode'] = debug_mode
    
    st.title("Tax Classification Model Fine-tuning")
    
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
        tab1, tab2 = st.tabs(["Fine-tune Model", "Get Predictions"])
        
        with tab1:
            
            # Existing fine-tuning code
            st.subheader("Sample Data")
            sample_data = get_sample_data()
            if sample_data:
                st.download_button(
                    label="Download Sample CSV",
                    data=sample_data,
                    file_name="sample_transactions.csv",
                    mime="text/csv",
                    help="Download a sample CSV file with the required format"
                )
            
            # File upload and validation
            st.subheader("Upload Your Data")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                # Read CSV with appropriate encoding
                df, error = read_csv_with_encoding(uploaded_file)
                
                if error:
                    st.error(error)
                    return
                
                if df is not None:
                    # Validate CSV structure
                    is_valid, missing_cols = validate_csv(df)
                    
                    if not is_valid:
                        st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
                        st.info("Please ensure your CSV has all required columns. You can download the sample CSV for reference.")
                        return
                    
                    st.success("CSV file validated successfully!")
                    
                    # Show data preview
                    st.subheader("Data Preview")
                    
                    # Calculate number of rows to show (3 from top and bottom if enough rows exist)
                    n_rows = 3
                    if len(df) > (n_rows * 2):
                        preview_df = pd.concat([df.head(n_rows), df.tail(n_rows)])
                        st.caption(f"Showing first and last {n_rows} rows of {len(df)} total rows")
                    else:
                        preview_df = df
                        st.caption(f"Showing all {len(df)} rows")
                    
                    st.dataframe(preview_df)
                    
                    if st.button("Start Fine-tuning"):
                        with st.spinner("Training..."):
                            temp_jsonl = None
                            try:
                                # Create temporary file
                                temp_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False).name
                                
                                # Convert to JSONL
                                if convert_csv_to_jsonl(uploaded_file, temp_jsonl):
                                    if debug_mode:
                                        st.write(f"Created temp file: {temp_jsonl}")
                                        st.write(f"File exists: {os.path.exists(temp_jsonl)}")
                                        with open(temp_jsonl, 'r') as f:
                                            st.write("First 200 chars:", f.read()[:200])
                                    
                                    # Start fine-tuning
                                    job_id, error = api.start_finetuning(temp_jsonl)
                                    
                                    if error:
                                        st.error(f"Error starting fine-tuning: {error}")
                                    else:
                                        # Monitor progress
                                        progress_placeholder = st.empty()
                                        while True:
                                            status = api.get_finetuning_status(job_id)
                                            if status["status"] == "completed":
                                                st.success("Fine-tuning completed successfully!")
                                                break
                                            elif status["status"] == "error":
                                                st.error(f"Fine-tuning failed: {status.get('error', 'Unknown error')}")
                                                break
                                            else:
                                                progress = status.get("progress", 0)
                                                progress_placeholder.progress(progress)
                                                time.sleep(10)
                                else:
                                    st.error("Failed to convert data. Please check your CSV file.")
                                    
                            except Exception as e:
                                st.error(f"Error during fine-tuning process: {str(e)}")
                                
                            finally:
                                # Temporarily disable cleanup for debugging
                                if debug_mode:
                                    st.write(f"Debug: Keeping temp file for inspection: {temp_jsonl}")
        
        with tab2:
            show_predictions_section(api)
            show_active_jobs(api)
    else:
        st.warning("Please enter your Together.ai API key to start")

if __name__ == "__main__":
    main()
