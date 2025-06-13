import json
import boto3
import pandas as pd
from typing import Optional, Tuple
from langdetect import detect
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
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def find_best_match(self, question: str) -> Tuple[str, float, str]:
        """Find the best matching question and its similarity score."""
        best_score = 0
        best_response = None
        question_lang = self.detect_language(question)
        
        for _, row in self.qa_dataset.iterrows():
            db_question = row['Question']
            db_lang = self.detect_language(db_question)
            
            # Only compare questions in the same language
            if db_lang == question_lang:
                score = calculate_similarity(question, db_question)
                if score > best_score:
                    best_score = score
                    best_response = row['Response']
        
        return best_response, best_score, question_lang

    def get_llm_response(self, question: str) -> Optional[str]:
        """Get response from AWS Bedrock LLM."""
        if not self.use_llm_fallback:
            return None
            
        try:            # Detect language and adjust prompt accordingly
            question_lang = self.detect_language(question)
            if question_lang == 'hi':
                prompt = f"""You are a helpful customer service assistant. Please provide a response in Hindi to the following question:

प्रश्न: {question}

उत्तर:"""
            else:
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
        best_response, score, lang = self.find_best_match(question)
        
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
