from abc import ABC, abstractmethod

class ModelAPI(ABC):
    """Abstract base class for ML model APIs"""
    
    @abstractmethod
    def upload_file(self, file_path):
        """Upload a file for training"""
        pass
    
    @abstractmethod
    def start_finetuning(self, jsonl_file):
        """Start fine-tuning process"""
        pass
    
    @abstractmethod
    def get_finetuning_status(self, job_id):
        """Get status of fine-tuning job"""
        pass
    
    @abstractmethod
    def get_models(self):
        """Get list of available models"""
        pass
    
    @abstractmethod
    def predict(self, model_id, transaction_details):
        """Make prediction using model"""
        pass
