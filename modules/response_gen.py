import json
import boto3
import pandas as pd
from typing import Optional, Tuple
from .utils import Config, load_qa_dataset, calculate_similarity

class ResponseGenerator:
    def __init__(self):
        self.config = Config()
        self.qa_dataset = load_qa_dataset()
        self.fuzzy_threshold = self.config.get('response', {}).get('fuzzy_match_threshold', 80)
        self.use_llm_fallback = self.config.get('response', {}).get('use_llm_fallback', True)
        
        if self.use_llm_fallback:
            self._setup_bedrock()

    def _setup_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            self.bedrock_client = boto3.client('bedrock-runtime')
            self.model_id = self.config.get('llm', {}).get('model_id', 'anthropic.claude-v2')
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
            self.use_llm_fallback = False

    def find_best_match(self, question: str) -> Tuple[str, float]:
        """Find the best matching question and its similarity score."""
        best_score = 0
        best_response = None
        
        for _, row in self.qa_dataset.iterrows():
            score = calculate_similarity(question, row['Question'])
            if score > best_score:
                best_score = score
                best_response = row['Response']
        
        return best_response, best_score

    def get_llm_response(self, question: str) -> Optional[str]:
        """Get response from AWS Bedrock LLM."""
        if not self.use_llm_fallback:
            return None
            
        try:
            prompt = f"""You are a helpful customer service assistant. Please provide a clear and concise response to the following question:

Question: {question}

Response:"""

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "prompt": prompt,
                    "max_tokens_to_sample": self.config.get('llm', {}).get('max_tokens', 500),
                    "temperature": self.config.get('llm', {}).get('temperature', 0.7),
                    "top_p": 0.9,
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body.get('completion', '')
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return None

    def get_response(self, question: str) -> Tuple[str, str, float]:
        """Get response for a given question.
        
        Returns:
            Tuple of (response text, source ['dataset' or 'llm'], confidence score)
        """
        # Try to find match in dataset
        best_response, score = self.find_best_match(question)
        
        # If good match found, return it
        if score >= self.fuzzy_threshold:
            return best_response, 'dataset', score
            
        # If no good match and LLM fallback enabled, try LLM
        if self.use_llm_fallback:
            llm_response = self.get_llm_response(question)
            if llm_response:
                return llm_response, 'llm', 100.0
        
        # If all else fails, return best match with low confidence
        return best_response or "I'm sorry, I couldn't find a suitable response.", 'dataset', score
