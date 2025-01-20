import json
import pandas as pd
import os

def format_prompt(row):
    """Format the input data as a prompt with JSON structure"""
    input_json = {
        "Description": row["Description"],
        "Category": row["Category"],
        "PaymentAccount": row["PaymentAccount"]
    }
    
    prompt = (
        "Given the following details in JSON format, extract the Vendor and TaxCategory "
        "and provide the response in JSON format. Input:\n"
        f"{json.dumps(input_json, indent=2)}\n"
        "Output the result as:\n"
        "{\n"
        "  \"Vendor\": \"<Vendor>\",\n"
        "  \"TaxCategory\": \"<TaxCategory>\"\n"
        "}"
    )
    
    return prompt

def format_completion(row):
    """Format the expected output as a JSON completion"""
    completion = {
        "Vendor": row["Vendor"],
        "TaxCategory": row["TaxCategory"]
    }
    return json.dumps(completion, indent=2)

def read_csv_with_encoding(file):
    """Try different encodings to read the CSV file"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            if hasattr(file, 'seek'):
                file.seek(0)  # Reset file pointer for each attempt
            df = pd.read_csv(file, encoding=encoding)
            return df, None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, str(e)
    
    return None, "Unable to read CSV file with any supported encoding"

def convert_csv_to_jsonl(input_file, output_file):
    """Convert CSV to JSONL format for fine-tuning"""
    try:
        # Read CSV with proper encoding
        df, error = read_csv_with_encoding(input_file)
        if error:
            print(f"Error reading CSV: {error}")
            return False
            
        if df is None or df.empty:
            print("DataFrame is empty")
            return False
            
        print(f"Columns found: {df.columns.tolist()}")  # Debug print
        print(f"First row: {df.iloc[0].to_dict()}")     # Debug print
        
        # Open output file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Process each row
            for _, row in df.iterrows():
                # Create training example
                example = {
                    "prompt": format_prompt(row),
                    "completion": format_completion(row)
                }
                
                # Write to JSONL file
                f.write(json.dumps(example) + '\n')
        
        # Debug - check output file
        print(f"Output file created: {os.path.exists(output_file)}")
        with open(output_file, 'r', encoding='utf-8') as f:
            print(f"First line of output: {f.readline()}")
            
        return True
        
    except Exception as e:
        print(f"Detailed error in convert_csv_to_jsonl: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    input_file = "activity-labeled.csv"
    output_file = "training_data.jsonl"
    if convert_csv_to_jsonl(input_file, output_file):
        print(f"Conversion complete. Output saved to {output_file}")
    else:
        print("Conversion failed") 