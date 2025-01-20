from model_api import ModelAPI
from predibase import Predibase,FinetuningConfig, DeploymentConfig
import time
import json

class PredibaseAPI(ModelAPI):
    """Predibase implementation"""
    def __init__(self, api_key, debug=False):
        self.debug = debug
        self.pb = Predibase(api_token=api_key)
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

    def predict(self, model_id, transaction_details):
        """Make prediction using Predibase SDK"""
        try:
            # Format input exactly like training data
            input_json = {
                "Description": transaction_details['Description'],
                "Category": transaction_details['Category'],
                "PaymentAccount": transaction_details['PaymentAccount']
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
            
            if self.debug:
                self._log(f"Prediction prompt: {prompt}")
            
            # Get prediction using deployments client
            lorax_client = self.pb.deployments.client("llama-3-1-8b-instruct")
            response = lorax_client.generate(
                prompt,
                adapter_id=model_id,  # This should be the adapter_id from fine-tuning
                max_new_tokens=100,
                temperature=0
            )
            
            if self.debug:
                self._log(f"Prediction response: {response}")
            
            return response.generated_text.strip(), None
            
        except Exception as e:
            self._log(f"Error in prediction: {str(e)}")
            return None, str(e)

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
