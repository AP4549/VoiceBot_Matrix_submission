import json
import boto3
import pandas as pd
import numpy as np
import os
import faiss
from typing import Optional, Tuple, List
from langdetect import detect
from botocore.exceptions import ClientError
import random # New import for random selection
from .utils import Config, load_qa_dataset

# Define paths for FAISS index and embeddings relative to the main project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'qa_embeddings.npy')
FAISS_INDEX_FILE = os.path.join(DATA_DIR, 'qa_faiss_index.bin')

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION = "us-west-2" # Ensure this matches your Bedrock region

class ResponseGeneratorRAG:
    _discourse_markers = [
        "Hmm, let me see...",
        "Okay, got it.",
        "Just a moment...",
        "Thinking...",
        "Right, let's find that out."
    ] # New: List of discourse markers

    def __init__(self):
        self.config = Config()
        self.qa_dataset = load_qa_dataset()
        self.semantic_similarity_threshold = self.config.get('response', {}).get('semantic_similarity_threshold', 0.50) # Adjusted threshold
        print(f"Debug: Semantic similarity threshold: {self.semantic_similarity_threshold}") # Added for debugging
        self.use_llm_fallback = self.config.get('response', {}).get('use_llm_fallback', True)
        self.top_k_retrieval = self.config.get('rag', {}).get('top_k_retrieval', 3) # Retrieve top 3 candidates
        
        self.bedrock_client = None # For LLM (Claude)
        self.embedding_client = None # For Embeddings (Titan)
        self.translate_client = None # For Translate
        self.faiss_index = None
        self.qa_embeddings_array = None

        self._setup_aws_clients() # Setup all AWS clients
        self._load_faiss_artifacts() # Load FAISS index and embeddings

    def _setup_aws_clients(self):
        """Initialize AWS Bedrock and Translate clients."""
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
            print(f"Debug: Claude LLM client initialized with model_id: {CLAUDE_MODEL_ID}")
            self.embedding_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
            print(f"Debug: Embedding client initialized with model_id: {EMBEDDING_MODEL_ID}")
            
            self.translate_client = boto3.client('translate', region_name=BEDROCK_REGION) # Initialize Translate client
            print(f"Debug: Translate client initialized")
        except Exception as e:
            print(f"Warning: Could not initialize AWS clients: {e}")
            self.use_llm_fallback = False

    def _load_faiss_artifacts(self):
        """Load FAISS index and embeddings."""
        try:
            print(f"Debug: Attempting to load FAISS index from: {FAISS_INDEX_FILE}")
            print(f"Debug: Attempting to load embeddings from: {EMBEDDINGS_FILE}")
            self.faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            self.qa_embeddings_array = np.load(EMBEDDINGS_FILE)
            print(f"Debug: FAISS index and embeddings loaded")
        except Exception as e:
            print(f"Warning: Could not load FAISS index and embeddings: {e}")
            self.use_llm_fallback = False # If FAISS fails, we can't do RAG effectively

    def _get_discourse_marker(self) -> str:
        """Randomly selects a discourse marker."""
        return random.choice(self._discourse_markers)

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            # Langdetect might fail for short/ambiguous texts
            lang = detect(text)
            # Map specific language codes if necessary, or just return as is
            # For example, 'hi' for Hindi, 'en' for English.
            return lang
        except:
            print("Warning: Could not detect language, defaulting to English.")
            return 'en'  # Default to English if detection fails

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Amazon Translate."""
        if not self.translate_client:
            print("Error: Translate client not initialized.")
            return text # Return original text if client not available
        if source_lang == target_lang: # No translation needed
            return text
        try:
            print(f"Debug: Attempting to translate from {source_lang} to {target_lang}")
            response = self.translate_client.translate_text(
                Text=text,
                SourceLanguageCode=source_lang,
                TargetLanguageCode=target_lang
            )
            translated_text = response.get('TranslatedText')
            if translated_text:
                print(f"Debug: Translation successful.")
                return translated_text
            else:
                print("Warning: Translate API returned empty text.")
                return text
        except ClientError as e:
            print(f"Error translating text (ClientError): {e}")
            if e.response and 'Error' in e.response:
                print(f"AWS Error Code: {e.response['Error'].get('Code')}")
                print(f"AWS Error Message: {e.response['Error'].get('Message')}")
            return text # Return original text on error
        except Exception as e:
            print(f"General error translating text: {e}")
            return text

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

    def find_best_matches_faiss(self, question: str) -> List[Tuple[str, float]]:
        """Find the best matching questions/responses using FAISS and semantic similarity.
        Returns a list of (response_text, confidence_score) tuples.
        """
        if self.faiss_index is None or self.qa_embeddings_array is None or self.qa_dataset is None:
            print("Warning: FAISS index or embeddings not loaded. Cannot perform FAISS search.")
            return []

        query_embedding = self._get_embedding(question)
        if query_embedding is None:
            print("Warning: Could not get embedding for question. Cannot perform FAISS search.")
            return []

        query_embedding = query_embedding.reshape(1, -1) # Reshape for FAISS search

        distances, indices = self.faiss_index.search(query_embedding, self.top_k_retrieval)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]

            # Calculate cosine similarity from L2 distance (assuming unit-normalized embeddings)
            similarity_score = 1 - (dist**2) / 2 
            confidence_score = max(0.0, min(100.0, similarity_score * 100)) # Clamp to 0-100
            
            response_text = self.qa_dataset.iloc[idx]['Response']
            results.append((response_text, confidence_score))

        return results

    def get_llm_response(self, question: str, context_documents: Optional[List[str]] = None) -> Optional[str]:
        """Get response from AWS Bedrock LLM, optionally using provided context documents."""
        if not self.use_llm_fallback:
            return None
            
        try:
            question_lang = self.detect_language(question)
            
            # Construct the prompt with context
            context_str = ""
            if context_documents:
                context_str = "\n\nContext:\n"
                for i, doc in enumerate(context_documents):
                    context_str += f"Document {i+1}: {doc}\n"
                context_str += "\n"

            # Create a general prompt template that incorporates language instruction and context
            # Make the language instruction prominent and instruct LLM to respond in target language and translate context if needed.
            full_prompt = f"""You are a helpful customer service assistant. Your primary goal is to answer the user's question in {question_lang} language.
If the provided context documents are in a different language, translate them internally to {question_lang} before answering.
Answer the user's question ONLY using the provided context. If the context is insufficient, state that you cannot answer based on the provided information.

{context_str}Question: {question}

Response:"""

            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            print(f"Debug: Full prompt sent to LLM: {full_prompt[:500]}...") # Debug full prompt

            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31", # Required for Claude 3
                    "messages": messages,
                    "max_tokens": self.config.get('llm', {}).get('max_tokens', 500),
                    "temperature": self.config.get('llm', {}).get('temperature', 0.7),
                    "top_p": 0.9,
                })
            )
            
            response_body = json.loads(response['body'].read())
            print(f"Debug: Raw LLM response_body: {response_body}") # Added for debugging LLM raw response

            llm_generated_text = ''
            if response_body.get('content'):
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        llm_generated_text = content_block['text']
            print(f"Debug: LLM generated text: {llm_generated_text[:100]}...") # Debug extracted LLM text
            return llm_generated_text
            
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
        """Get response for a given question using FAISS and LLM fallback.
        
        Returns:
            Tuple of (response text, source ['dataset (FAISS)' or 'llm (augmented)' or 'llm'], confidence score)
        """
        # Find best semantic matches from dataset using FAISS
        top_k_matches = self.find_best_matches_faiss(question)
        
        best_dataset_response = None
        best_dataset_score = 0.0

        if top_k_matches:
            # The first match is the best one from FAISS
            best_dataset_response, best_dataset_score = top_k_matches[0]

        # If the best FAISS match is confident enough, use it directly from the dataset
        if best_dataset_response and best_dataset_score >= self.semantic_similarity_threshold:
            final_response_text = best_dataset_response
            final_source = 'dataset (FAISS)'
            final_confidence = best_dataset_score
            
        # If no good semantic match, but LLM fallback is enabled, use LLM (potentially with retrieved context)
        elif self.use_llm_fallback:
            context_documents = [match[0] for match in top_k_matches if match[0]] # Extract just the text responses
            
            llm_response = self.get_llm_response(question, context_documents=context_documents)
            
            if llm_response:
                final_response_text = llm_response
                final_source = 'llm (augmented with FAISS)' if context_documents else 'llm (no context)'
                final_confidence = 100.0 # LLM confidence is typically 100% if it returns a response
            else:
                # If LLM didn't generate a response, fall back to best dataset match (low confidence)
                final_response_text = best_dataset_response or "I'm sorry, I couldn't find a suitable response."
                final_source = 'dataset (low confidence)'
                final_confidence = best_dataset_score
        else:
            # If LLM fallback is not enabled, return best dataset match (even if low confidence)
            final_response_text = best_dataset_response or "I'm sorry, I couldn't find a suitable response."
            final_source = 'dataset (low confidence)'
            final_confidence = best_dataset_score

        # Post-processing: Ensure final response language matches question language
        question_lang = self.detect_language(question)
        response_lang = self.detect_language(final_response_text)

        if question_lang != response_lang:
            print(f"Debug: Mismatch in languages. Question: {question_lang}, Response: {response_lang}. Attempting translation.")
            translated_response = self._translate_text(final_response_text, response_lang, question_lang)
            if translated_response != final_response_text: # Check if translation actually occurred
                final_response_text = translated_response
                final_source = f"{final_source} (translated)"
        
        # Add conversational nuance
        conversational_prefix = self._get_discourse_marker() + " "
        final_response_text = conversational_prefix + final_response_text

        return final_response_text, final_source, final_confidence 