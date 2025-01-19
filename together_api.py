import time
import json
from together import Together

class TogetherAPI:
    def __init__(self, api_key, debug=False):
        self.debug = debug
        self.client = Together(api_key=api_key)
        if debug:
            self._log("Initialized Together client")
    
    def _log(self, message):
        """Debug logging helper"""
        if self.debug:
            print(f"TogetherAPI: {message}")
    
    def upload_file(self, file_path):
        """Upload a file using Together SDK"""
        try:
            if self.debug:
                self._log(f"Attempting to upload file: {file_path}")
            
            response = self.client.files.upload(
                file=file_path,
                purpose="fine-tune"
            )
            
            if self.debug:
                self._log(f"Upload response: {response}")
                
            return response.id, None
            
        except Exception as e:
            self._log(f"Error uploading file: {str(e)}")
            return None, str(e)
    
    def start_finetuning(self, jsonl_file):
        """Start fine-tuning using Together SDK"""
        try:
            # First upload the file
            file_id, error = self.upload_file(jsonl_file)
            if error:
                return None, f"File upload failed: {error}"

            if self.debug:
                self._log(f"File uploaded successfully with ID: {file_id}")
            
            # Start fine-tuning with Meta-Llama-3
            response = self.client.fine_tuning.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",  # Changed to Meta-Llama-3
                training_file=file_id,                
                n_epochs=4,
                batch_size=16
            )
            
            if self.debug:
                self._log(f"Fine-tune response: {response}")
            
            return response.id, None
            
        except Exception as e:
            self._log(f"Error in fine-tuning: {str(e)}")
            return None, str(e)
    
    def get_finetuning_status(self, job_id):
        """Get fine-tuning status using Together SDK"""
        try:
            response = self.client.fine_tuning.retrieve(job_id)
            return response.model_dump()
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_models(self):
        """Get list of fine-tuned models using Together SDK"""
        try:
            response = self.client.fine_tuning.list()
            # The response directly contains the list of fine-tunes
            return response.data, None  # Changed from response.fine_tunes to response.data
            
        except Exception as e:
            self._log(f"Error getting models: {str(e)}")
            return None, str(e)
    
    def predict(self, model_id, transaction_details):
        """Get prediction using Together SDK"""
        try:
            # Get model details to get the full output_name
            response = self.client.fine_tuning.retrieve(model_id)
            if not response.output_name:
                return None, "Model output name not found"
            
            if self.debug:
                self._log(f"Using model: {response.output_name}")
            
            # Format input exactly like training data
            transaction = (
                f"Description: {transaction_details['Description']}\n"
                f"Category: {transaction_details['Category']}\n"
                f"PaymentAccount: {transaction_details['PaymentAccount']}"
            )
            
            # Wrap in instruction format
            prompt = (
                "[INST] Given this transaction, extract the vendor name and determine "
                "the appropriate tax category. Format the response as a JSON object "
                f"with 'Vendor' and 'TaxCategory' fields:\n\n{transaction} [/INST]"
            )
            
            if self.debug:
                self._log(f"Prompt: {prompt}")
            
            # Make prediction using the full model name
            response = self.client.completions.create(
                model=response.output_name,
                prompt=prompt,
                temperature=0,
                max_tokens=100,
                stop=["[INST]"]  # Stop at next instruction
            )
            
            return response.choices[0].text.strip(), None
            
        except Exception as e:
            self._log(f"Error in prediction: {str(e)}")
            return None, str(e) 