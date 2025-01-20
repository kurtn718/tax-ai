from enum import Enum
from model_api import ModelAPI
#from together_api import TogetherAPI
#from predibase_api import PredibaseAPI
#from ../predibase_api.py import PredibaseAPI
import predibasellm_api

class ModelProvider(Enum):
    TOGETHER = "together"
    PREDIBASE = "predibase"

class ModelFactory:
    @staticmethod
    def create(provider: ModelProvider, api_key: str, debug: bool = False, tax_categories=None) -> ModelAPI:
#        if provider == ModelProvider.TOGETHER:
#            return TogetherAPI(api_key, debug)
#        elif provider == ModelProvider.PREDIBASE:
#            return PredibaseAPI(api_key, debug)
#        else:
#            raise ValueError(f"Unknown provider: {provider}") 
        return predibasellm_api.PredibaseAPI(api_key, debug, tax_categories=tax_categories)
