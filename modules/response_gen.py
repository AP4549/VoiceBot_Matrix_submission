import json
import boto3
import pandas as pd
from typing import Optional, Tuple
from langdetect import detect
from botocore.exceptions import ClientError
from .utils import Config, load_qa_dataset, calculate_similarity
import faiss # New import for FAISS
import numpy as np # New import for NumPy
import os # New import for path handling

# Define paths for FAISS index and embeddings relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'qa_embeddings.npy')
FAISS_INDEX_FILE = os.path.join(DATA_DIR, 'qa_faiss_index.bin')

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0" # Titan Embeddings model
BEDROCK_REGION = "us-west-2" # Ensure this matches your Bedrock region

class ResponseGenerator:
    def __init__(self):
        self.config = Config()
        self.qa_dataset = load_qa_dataset()
        # self.fuzzy_threshold = self.config.get('response', {}).get('fuzzy_match_threshold', 80) # Old fuzzy threshold
        self.semantic_similarity_threshold = self.config.get('response', {}).get('semantic_similarity_threshold', 0.50) # New threshold for FAISS, adjusted
        self.use_llm_fallback = self.config.get('response', {}).get('use_llm_fallback', True)
        
        self.bedrock_client = None # For LLM (Claude)
        self.embedding_client = None # For Embeddings (Titan)
        self.faiss_index = None
        self.qa_embeddings_array = None

        self._setup_bedrock() # Setup for LLM (Claude)
        self._setup_embedding_client() # New: Setup for Embedding model (Titan)
        self._load_faiss_artifacts() # New: Load FAISS index and embeddings

    def _setup_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            self.bedrock_client = boto3.client('bedrock-runtime')
            self.model_id = self.config.get('llm', {}).get('model_id', 'anthropic.claude-3-sonnet-20240229-v1:0')
            print(f"Debug: Bedrock client initialized with model_id: {self.model_id}")
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
            self.use_llm_fallback = False

    def _setup_embedding_client(self):
        """Initialize AWS Embedding client."""
        try:
            self.embedding_client = boto3.client('bedrock-runtime')
            print(f"Debug: Embedding client initialized")
        except Exception as e:
            print(f"Warning: Could not initialize Embedding client: {e}")
            self.use_llm_fallback = False

    def _load_faiss_artifacts(self):
        """Load FAISS index and embeddings."""
        try:
            print(f"Debug: Attempting to load FAISS index from: {FAISS_INDEX_FILE}") # Added for debugging
            print(f"Debug: Attempting to load embeddings from: {EMBEDDINGS_FILE}") # Added for debugging
            self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            self.qa_embeddings_array = np.load(EMBEDDINGS_FILE)
            print(f"Debug: FAISS index and embeddings loaded")
        except Exception as e:
            print(f"Warning: Could not load FAISS index and embeddings: {e}")
            self.use_llm_fallback = False

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Gets an embedding for the given text using Bedrock Titan Text Embeddings."""
        if not self.embedding_client:
            print("Error: Embedding client not initialized.")
            return None
        try:
            body = json.dumps({"inputText": text})
            response = self.embedding_client.invoke_model(
                body=body,
                modelId=EMBEDDING_MODEL_ID,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            embedding = response_body.get('embedding')
            if not embedding:
                print("Error: Embedding not found in Bedrock response.")
                return None
            return np.array(embedding).astype('float32')
        except ClientError as e:
            print(f"Error getting embedding from Bedrock (ClientError): {e}")
            if e.response and 'Error' in e.response:
                print(f"AWS Error Code: {e.response['Error'].get('Code')}")
                print(f"AWS Error Message: {e.response['Error'].get('Message')}")
            return None
        except Exception as e:
            print(f"General error getting embedding: {e}")
            return None

    def find_best_match(self, question: str) -> Tuple[Optional[str], float, str]:
        """Find the best matching question using FAISS and semantic similarity."""
        question_lang = self.detect_language(question)

        if self.faiss_index is None or self.qa_embeddings_array is None or self.qa_dataset is None:
            print("Warning: FAISS index or embeddings not loaded. Falling back to simple response.")
            return None, 0.0, question_lang

        query_embedding = self._get_embedding(question)
        if query_embedding is None:
            print("Warning: Could not get embedding for question. Falling back to simple response.")
            return None, 0.0, question_lang

        # Reshape for FAISS search (1 query vector)
        query_embedding = query_embedding.reshape(1, -1)

        # Search the FAISS index for the top K (e.g., 1) nearest neighbors
        k = 1  # You can increase K to retrieve multiple similar items
        distances, indices = self.faiss_index.search(query_embedding, k)

        # FAISS returns L2 distance. Convert to a similarity score if needed. Closer to 0 is more similar.
        # A common way to get a 'similarity' from L2 distance is 1 / (1 + distance), or normalize embeddings and use dot product.
        # For simplicity, we'll use a threshold directly on distance for now, or use normalized dot product if embeddings are normalized.
        
        # Assuming embeddings are normalized for cosine similarity (dot product = cosine similarity)
        # For L2, lower distance means higher similarity. We need to convert.
        # If embeddings are normalized, L2 distance (d) is related to cosine similarity (s) by d^2 = 2 * (1 - s)
        # So, s = 1 - (d^2 / 2). A score of 1 means perfect match (0 distance).

        best_distance = distances[0][0]
        best_match_index = indices[0][0]

        # Calculate cosine similarity (more intuitive for 'confidence')
        # Make sure embeddings are normalized when generated if you want direct dot product = cosine sim
        # For L2 distance: similarity = 1 - (distance / max_possible_distance) or 1 / (1 + distance)
        # Let's use a simple inverted distance for score as a proxy for confidence for now.
        # Or, ideally, compute cosine similarity between query_embedding and retrieved_embedding.
        
        # For now, let's just return a score based on how close the distance is to 0.
        # A very simple proxy: score = max(0, 1 - (best_distance / some_max_expected_distance))
        # A better proxy using L2 distance properties if embeddings are normalized:
        similarity_score = 1 - (best_distance**2) / 2 # This holds if embeddings are unit normalized
        
        # Clamp the score between 0 and 100 for confidence display
        confidence_score = max(0.0, min(100.0, similarity_score * 100))

        best_response = self.qa_dataset.iloc[best_match_index]['Response']
        return best_response, confidence_score, question_lang

    def get_llm_response(self, question: str, context: Optional[str] = None) -> Optional[str]:
        """Get response from AWS Bedrock LLM, optionally using provided context."""
        if not self.use_llm_fallback:
            return None
            
        try:
            question_lang = self.detect_language(question)
            
            user_content = f"Question: {question}"
            if context:
                user_content = f"I have the following information: {context}. Based on this information, {question}"

            if question_lang == 'hi':
                messages = [
                    {
                        "role": "user",
                        "content": f"आप एक सहायक ग्राहक सेवा सहायक हैं। कृपया निम्नलिखित प्रश्न का हिंदी में उत्तर दें:\n\n{user_content}\n\nउत्तर:"
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": f"You are a helpful customer service assistant. Please provide a clear and concise response to the following question:\n\n{user_content}\n\nResponse:"
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
        # Try to find match in dataset using FAISS
        best_response_from_dataset, score, lang = self.find_best_match(question)
        
        # If good semantic match found, return it directly from the dataset
        if score >= self.semantic_similarity_threshold:
            return best_response_from_dataset, 'dataset', score
            
        # If no good semantic match, but LLM fallback is enabled, use LLM (potentially with retrieved context)
        if self.use_llm_fallback:
            context_for_llm = best_response_from_dataset if best_response_from_dataset and score > 0 else None
            llm_response = self.get_llm_response(question, context=context_for_llm)
            if llm_response:
                return llm_response, 'llm (augmented)' if context_for_llm else 'llm', 100.0
        
        # If all else fails (no good dataset match and LLM didn't generate a response), 
        # return the best dataset match found (even if low confidence) or a default message.
        return best_response_from_dataset or "I'm sorry, I couldn't find a suitable response.", 'dataset (low confidence)', score
