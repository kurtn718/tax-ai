import requests
from model_api import ModelAPI
from predibase import Predibase,FinetuningConfig, DeploymentConfig
import time
import json
import os
from config import DEFAULT_TAX_CATEGORIES, WORD_CORRECTIONS

class PredibaseAPI(ModelAPI):
    """Predibase implementation using REST API"""
    def __init__(self, api_key, debug=False, tax_categories=None):
        self.debug = debug
        self.api_key = api_key
        self.tenant_id = os.getenv('PREDIBASE_TENANT_ID')
        self.base_url = "https://serving.app.predibase.com"
        self.tax_categories = tax_categories or DEFAULT_TAX_CATEGORIES
        if debug:
            self._log("Initialized Predibase client")

    def _log(self, message):
        """Debug logging helper"""
        if self.debug:
            print(f"PredibaseAPI: {message}")

    def upload_file(self, file_path):
        # Make this a no-op
        return None, None

    def start_finetuning(self, jsonl_file):
        """Start fine-tuning using Predibase SDK"""
        try:
            # Create dataset using pb.datasets
            dataset = self.pb.datasets.from_file(
                jsonl_file,
                name=f"tax_dataset_{int(time.time())}"  # Unique name
            )

            # Create repository using pb.repos
            repo = self.pb.repos.create(
                name=f"tax-classifier-{int(time.time())}", 
                description="Tax Classification Model",
                exists_ok=True
            )

            # Create adapter using pb.adapters
            adapter = self.pb.adapters.create(
               config=FinetuningConfig(
                    base_model="llama-3-1-8b-instruct",  # Using llama-3 model,
                    batch_size=4,
                    learning_rate=1e-5,
                    num_epochs=4
                ),
                dataset=dataset,
                repo=repo,
                description="Tax classification model"
            )
            
            if self.debug:
                self._log(f"Fine-tune adapter created: {adapter}")
            
            return adapter, None
            
        except Exception as e:
            self._log(f"Error in fine-tuning: {str(e)}")
            return None, str(e)

    def get_finetuning_status(self, job_id):
        """Get fine-tuning status using Predibase SDK"""
        # Not implemented
        return None, None

    def get_models(self):
        """Get list of fine-tuned models using Predibase SDK"""
        try:
            repos = self.pb.repos.list()

            models = []
            for repo in repos:
                repo = self.pb.repos.get(repo.uuid)

                # Get all versions for this repo
                versions = repo.all_versions
                for version in versions:
                    if not version.archived:
                        models.append({
                            "id": repo.uuid,
                            "tag": version.tag,
                            "output_name": f"{repo.name} v{version.tag}",
                            "created_at": version.created_at,
                            "status": None
                        })
                    
            return models, None
            
        except Exception as e:
            self._log(f"Error getting models: {str(e)}")
            return None, str(e)

    def predict(self, model_id, transactions):
        """Make prediction using Predibase REST API"""
        all_results = []
        
        for txn in transactions:
            try:
                input_json = {
                    "Description": txn['Description'],
                    "Category": txn['Category']
                }
                
                # Format word corrections for prompt
                corrections_str = json.dumps(WORD_CORRECTIONS, indent=2)
                # Format tax categories for prompt
                categories_str = json.dumps(self.tax_categories, indent=2)
                
                prompt = (
                    "Given the following transaction details in JSON format, extract the Vendor and TaxCategory "
                    "and provide the response as a single JSON object.\n\n"
                    "You can strip things like GPay and City / State. Note that the City might not be separated by a space from the "
                    "actual vendor name (i.e. given the description GPay LENNYSUBSATLANTA GA - we would only want LENNY SUB). "
                    "If you recognize a word that is compressed you can expand it.\n\n"
                    f"Words that I would like corrected:\n{corrections_str}\n\n"
                    f"Tax Categories are:\n{categories_str}\n\n"
                    f"Input: {json.dumps(input_json, indent=2)}\n\n"
                    "Output must be a single JSON object with Vendor and TaxCategory. "
                    "Only output valid JSON object. No other text."
                )
                
                # Store last prompt for debugging
                self.last_prompt = prompt
                
                if self.debug:
                    self._log(f"Prediction prompt: {prompt}")

                # Make REST API call
                url = f"{self.base_url}/{self.tenant_id}/deployments/v2/llms/llama-3-3-70b-instruct/generate"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 1000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                }

                if self.debug:
                    self._log(f"Making API call to: {url}")
                    
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()  # Raise exception for bad status codes
                
                result = response.json()
                raw_text = result.get('generated_text', '').strip()
                
                if self.debug:
                    self._log(f"Raw API response: {result}")
                
                # Try to find JSON by looking for opening/closing brackets
                try:
                    start_idx = raw_text.find('[')
                    end_idx = raw_text.rfind(']') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_text = raw_text[start_idx:end_idx]
                    else:
                        start_idx = raw_text.find('{')
                        end_idx = raw_text.rfind('}') + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_text = raw_text[start_idx:end_idx]
                        else:
                            json_text = raw_text
                    
                    # Validate it's valid JSON
                    json.loads(json_text)  # This will raise an error if not valid JSON
                    
                    if self.debug:
                        self._log(f"Raw response: {raw_text}")
                        self._log(f"Extracted JSON: {json_text}")
                    
                    all_results.append(json_text)
                    
                except json.JSONDecodeError:
                    if self.debug:
                        self._log(f"Failed to extract valid JSON from: {raw_text}")
                    all_results.append({"error": "Failed to get valid JSON response"})
                
            except Exception as e:
                self._log(f"Error processing transaction: {str(e)}")
                all_results.append({"error": str(e)})
        
        return json.dumps(all_results), None

    def get_active_jobs(self):
        """Get list of active fine-tuning jobs"""
        try:
            # Get all active adapters
            active_jobs = self.pb.adapters.list_active()
            
            jobs = []
            for job in active_jobs:
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "status": job.status,
                    "progress": job.progress if hasattr(job, 'progress') else 0,
                    "created_at": job.created_at if hasattr(job, 'created_at') else None
                })
                
            if self.debug:
                self._log(f"Found {len(jobs)} active jobs")
                
            return jobs, None
            
        except Exception as e:
            self._log(f"Error getting active jobs: {str(e)}")
            return None, str(e)
