import os
import yaml
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, Any
from fuzzywuzzy import fuzz

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Load environment variables from .env
        load_dotenv()
        
        # Load config from yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)

def clean_text(text: str) -> str:
    """Clean and normalize text for better matching."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove punctuation if needed
    # text = text.translate(str.maketrans("", "", string.punctuation))
    
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts."""
    # Clean both texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Use token sort ratio for better matching regardless of word order
    return fuzz.token_sort_ratio(text1, text2)

def load_qa_dataset() -> pd.DataFrame:
    """Load the QA dataset from CSV."""
    config = Config()
    qa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.get('qa_csv'))
    
    try:
        df = pd.read_csv(qa_path)
        required_columns = ['Question', 'Response']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    except Exception as e:
        print(f"Error loading QA dataset: {e}")
        return pd.DataFrame(columns=['Question', 'Response'])

def get_aws_credentials() -> Dict[str, str]:
    """Get AWS credentials from environment variables."""
    credentials = {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'aws_session_token': os.getenv('AWS_SESSION_TOKEN'),
        'region_name': Config().get('region')
    }
    
    # Remove None values
    return {k: v for k, v in credentials.items() if v is not None}
