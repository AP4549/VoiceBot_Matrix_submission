import json
import boto3
import pandas as pd
from typing import Optional, Tuple
from langdetect import detect
from botocore.exceptions import ClientError
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
            self.model_id = self.config.get('llm', {}).get('model_id', 'anthropic.claude-3-sonnet-20240229-v1:0')
            print(f"Debug: Bedrock client initialized with model_id: {self.model_id}")
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
                messages = [
                    {
                        "role": "user",
                        "content": f"आप एक सहायक ग्राहक सेवा सहायक हैं। कृपया निम्नलिखित प्रश्न का हिंदी में उत्तर दें:\n\nप्रश्न: {question}\n\nउत्तर:"
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": f"You are a helpful customer service assistant. Please provide a clear and concise response to the following question:\n\nQuestion: {question}\n\nResponse:"
                    }
                ]

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31", # Required for Claude 3
                    "messages": messages,
                    "max_tokens": self.config.get('llm', {}).get('max_tokens', 500),
                    "temperature": self.config.get('llm', {}).get('temperature', 0.7),
                    "top_p": 0.9,
                })
            )
            
            response_body = json.loads(response['body'].read())
            # Extract content from the Messages API response
            if response_body.get('content'):
                # Find the first text content block
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        return content_block['text']
            return '' # Return empty string if no text content found
            
        except ClientError as e:
            print(f"Error getting LLM response from Bedrock (ClientError): {e}")
            if e.response and 'Error' in e.response:
                print(f"AWS Error Code: {e.response['Error'].get('Code')}")
                print(f"AWS Error Message: {e.response['Error'].get('Message')}")
            return None    
        except Exception as e:
            print(f"Error getting LLM response (General Exception): {e}")
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
