import json
import boto3
import pandas as pd
import numpy as np
import os
import faiss
import logging
from typing import Optional, Tuple, List, Dict
from langdetect import detect
from botocore.exceptions import ClientError
import random
from .utils import Config, load_qa_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from .supabase_client import SupabaseManager
from .vector_memory import VectorMemoryManager
from .vector_memory import VectorMemoryManager

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION = "us-west-2"

# Configure logging
logger = logging.getLogger(__name__)

class ResponseGeneratorRAGB:
    _discourse_markers = [
        "Hmm, let me see...",
        "Okay, got it.",
        "Just a moment...",
        "Thinking...",
        "Right, let's find that out."
    ]

    def __init__(self, faiss_index_path: str, embeddings_file_path: str):
        self.config = Config()
        self.qa_dataset = load_qa_dataset()
        self.semantic_similarity_threshold = self.config.get('response', {}).get('semantic_similarity_threshold', 0.50)
        self.use_llm_fallback = self.config.get('response', {}).get('use_llm_fallback', True)
        self.top_k_retrieval = self.config.get('rag', {}).get('top_k_retrieval', 3)
        self.max_context_window = self.config.get('response', {}).get('max_context_window', 5)
        self.use_vector_memory = self.config.get('memory', {}).get('use_vector_memory', True)

        self.bedrock_client = None
        self.embedding_client = None
        self.translate_client = None
        self.faiss_index = None
        self.qa_embeddings_array = None
        self.supabase = SupabaseManager()

        self._setup_aws_clients()
        self._load_faiss_artifacts(faiss_index_path, embeddings_file_path)
        
        # Initialize vector memory if enabled
        if self.use_vector_memory and self.embedding_client:
            try:
                self.vector_memory = VectorMemoryManager(
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    bedrock_client=self.embedding_client,
                    index_path="Data/memory_index.faiss",
                    metadata_path="Data/memory_metadata.json"
                )
                logger.info("Vector memory manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize vector memory: {e}")
                self.use_vector_memory = False

    def _setup_aws_clients(self):
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
            self.embedding_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
            self.translate_client = boto3.client('translate', region_name=BEDROCK_REGION)
        except Exception as e:
            print(f"Warning: Could not initialize AWS clients: {e}")
            self.use_llm_fallback = False

    def _load_faiss_artifacts(self, faiss_index_path: str, embeddings_file_path: str):
        try:
            self.faiss_index = faiss.read_index(faiss_index_path)
            self.qa_embeddings_array = np.load(embeddings_file_path)
        except Exception as e:
            print(f"Warning: Could not load FAISS index and embeddings: {e}")
            self.use_llm_fallback = False

    def _get_discourse_marker(self) -> str:
        return random.choice(self._discourse_markers)

    def detect_language(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang not in ['en', 'hi']:
                return 'hi'
            return lang
        except:
            return 'hi'

    def _is_hinglish(self, text: str) -> bool:
        if not text:
            return False
        devanagari_count = sum('\u0900' <= c <= '\u097F' for c in text)
        latin_count = sum('a' <= c.lower() <= 'z' for c in text if c.isalpha())
        return devanagari_count == 0 and latin_count > 5

    def polish_hinglish(self, raw_text: str) -> str:
        prompt = f"""You are a helpful assistant that speaks in natural Hinglish (Hindi written in Roman script).
Improve the readability of this sentence by making it more conversational, lowercase, and easy to understand.
Avoid awkward capitalizations or phonetic symbols.

Text: {raw_text}
Improved Hinglish:"""

        try:
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3,
                })
            )
            content_blocks = json.loads(response["body"].read()).get("content", [])
            return content_blocks[0]["text"] if content_blocks and content_blocks[0].get("type") == "text" else raw_text
        except Exception as e:
            print(f"Warning: Could not polish Hinglish: {e}")
            return raw_text

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source language to target language using AWS Translate"""
        if not self.translate_client:
            return text
        if source_lang == target_lang:
            return text
        try:
            response = self.translate_client.translate_text(
                Text=text,
                SourceLanguageCode=source_lang,
                TargetLanguageCode=target_lang
            )
            return response.get('TranslatedText', text)
        except Exception as e:
            print(f"Warning: Translation failed: {e}")
            return text

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get single text embedding"""
        if not self.embedding_client:
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
            return np.array(embedding).astype('float32') if embedding else None
        except Exception as e:
            print(f"Warning: Could not get embedding: {e}")
            return None

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get multiple text embeddings (batch processing)"""
        try:
            embeddings = []
            for text in texts:
                embedding = self._get_embedding(text)
                if embedding is not None:
                    embeddings.append(embedding)
            return np.array(embeddings) if embeddings else None
        except Exception as e:
            print(f"Warning: Could not get embeddings: {e}")
            raise

    def find_best_matches_faiss(self, question: str) -> List[Tuple[str, float]]:
        """Find best matches using FAISS index with proper confidence scoring"""
        if self.faiss_index is None or self.qa_embeddings_array is None or self.qa_dataset is None:
            return []
        
        query_embedding = self._get_embedding(question)
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, self.top_k_retrieval)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            # Convert distance to similarity score
            similarity_score = 1 - (dist**2) / 2 
            confidence_score = max(0.0, min(100.0, similarity_score * 100))
            
            try:
                response_text = self.qa_dataset.iloc[idx]['Response']
                results.append((response_text, confidence_score))
            except Exception as e:
                print(f"Error accessing dataset at index {idx}: {e}")
                continue
                
        return results

    def enhance_with_llm(self, question: str, dataset_answer: str) -> str:
        """Use the LLM to rewrite/enhance the dataset answer for clarity, completeness, and empathy."""
        try:
            prompt = f"""
You are a helpful, empathetic customer service assistant. The following is a user's question and a draft answer from a dataset. Please rewrite or elaborate the answer to be clear, complete, empathetic, and concise (limit your enhanced answer to 50 words). Use a respectful and friendly tone. If the answer is already good, you may polish it slightly for clarity and empathy.

Question: {question}
Draft Answer: {dataset_answer}

Enhanced Answer (be concise, limit to 50 words):
"""
            messages = [{"role": "user", "content": prompt}]
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": messages,
                    "max_tokens": self.config.get('llm', {}).get('max_tokens', 100),
                    "temperature": self.config.get('llm', {}).get('temperature', 0.7),
                    "top_p": 0.9,
                })
            )
            response_body = json.loads(response['body'].read())
            enhanced_text = ''
            if response_body.get('content'):
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        enhanced_text = content_block['text']
            return enhanced_text.strip() if enhanced_text else dataset_answer
        except Exception as e:
            print(f"Warning: Could not enhance dataset answer with LLM: {e}")
            return dataset_answer

    def get_llm_response(self, question: str, context_documents: Optional[List[str]] = None, conversation_context: str = None) -> Optional[str]:
        """Get LLM response with optional context documents and conversation history"""
        if not self.use_llm_fallback:
            return None
        try:
            context_str = ""
            if context_documents:
                context_str = "\n\nContext:\n" + "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_documents)]) + "\n"

            # Add conversation context if available
            conversation_str = ""
            if conversation_context:
                conversation_str = f"\n\nPREVIOUS CONVERSATION HISTORY (CRITICAL - YOU MUST REFERENCE THIS WHEN ANSWERING):\n{conversation_context}\n"

            full_prompt = f"""You are a helpful, empathetic customer service assistant. Your primary goal is to answer the user's question in English (or in the original language if the question is not in English) in a professional, friendly, and empathetic tone. 

IMPORTANT INSTRUCTIONS:
1. You MUST reference relevant information from the conversation history when responding.
2. If the user refers to something mentioned earlier, acknowledge it explicitly and provide continuity.
3. Always maintain context awareness across multiple turns of conversation.
4. If the provided context documents are in a different language, translate them internally to English before answering.
5. Answer the user's question ONLY using the provided context and conversation history.
6. If the context is insufficient, state that you cannot answer based on the provided information.
7. Use respectful phrases (e.g. "bataiye", "boliye", "haan grahak ji") and avoid informal phrases (like "haa bhai").
8. NEVER forget previous parts of the conversation. ALWAYS refer back to previous exchanges when relevant.
9. If the user asks about something mentioned in a previous message, ALWAYS acknowledge that you remember it.
10. PRIORITIZE maintaining conversation continuity over being overly friendly.

{context_str}{conversation_str}Question: {question}\n\nResponse:"""

            # Debug logging to verify context is being sent to LLM
            logger.debug(f"LLM Prompt with Context:\n{full_prompt}")
            
            messages = [{"role": "user", "content": full_prompt}]
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": messages,
                    "max_tokens": self.config.get('llm', {}).get('max_tokens', 500),
                    "temperature": self.config.get('llm', {}).get('temperature', 0.7),
                    "top_p": 0.9,
                })
            )
            response_body = json.loads(response['body'].read())
            llm_generated_text = ''
            if response_body.get('content'):
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        llm_generated_text = content_block['text']
            return llm_generated_text
        except Exception as e:
            print(f"Warning: LLM response failed: {e}")
            return None

    def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from AWS Bedrock Claude model (alternative method)"""
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"I understand I am a bank's voice assistant. {system_prompt}"
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "temperature": 0.7,
                "top_k": 250,
                "top_p": 0.999
            })
            
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=body
            )
            response_body = json.loads(response.get('body').read())
            
            # Check response format and extract content
            if isinstance(response_body, dict):
                if 'content' in response_body and isinstance(response_body['content'], list):
                    for content_block in response_body['content']:
                        if content_block.get('type') == 'text':
                            return content_block['text']
                elif 'completion' in response_body:
                    return response_body['completion']
                elif 'choices' in response_body and len(response_body['choices']) > 0:
                    return response_body['choices'][0].get('message', {}).get('content', '')
            raise ValueError("Unexpected response format from LLM")
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            raise
            
    def get_conversation_context(self, user_id: str, query: str = None, access_token: str = None) -> str:
        """Get conversations from memory storage and format them as context.
        
        This method uses vector-based retrieval when possible, falling back to recency-based
        retrieval if vector memory is not available or the query is None.
        """
        try:
            # Try using vector memory for semantic search if enabled and query is provided
            if self.use_vector_memory and self.vector_memory and query:
                logger.info(f"Using vector memory retrieval for user {user_id}")
                
                # Use hybrid retrieval (combining semantic and recency)
                memories = self.vector_memory.hybrid_retrieve(
                    query=query, 
                    user_id=user_id,
                    recent_limit=3,  # Get 3 recent conversations
                    semantic_limit=3  # Get 3 semantically similar conversations
                )
                
                if memories:
                    # Format memories as context
                    context = self.vector_memory.format_memories_as_context(memories)
                    logger.info(f"Retrieved {len(memories)} memories using vector similarity")
                    return context
            
            # Fall back to recency-based retrieval
            logger.info(f"Using recency-based memory retrieval for user {user_id}")
            
            # Get last 6 conversations from Supabase
            conversations = self.supabase.get_recent_conversations(
                user_id=user_id,
                access_token=access_token,
                limit=6
            )
            
            # Format conversations as context
            context = "LONG-TERM CONVERSATION HISTORY (CRITICAL - YOU MUST REFERENCE THIS WHEN ANSWERING):\n"
            for i, conv in enumerate(conversations):
                context += f"Exchange {i+1}:\nHuman: {conv['message']}\nAssistant: {conv['response']}\n\n"
            return context.strip()
        except Exception as e:
            logger.warning(f"Could not retrieve conversation history: {str(e)}")
            return ""
            
    def get_response(self, question: str, context: str = None, chat_history: List[List[str]] = None, user_id: str = None, access_token: str = None) -> Tuple[str, str, float]:
        """
        Main method that combines all functionality from both files:
        - Language detection and translation
        - FAISS search with proper confidence scoring
        - Answer enhancement for non-Hindi questions
        - Vector memory retrieval for semantically relevant conversation history
        - Conversation context support from Supabase and in-memory chat history
        - Hinglish transliteration and polishing
        """
        try:
            # Step 1: Detect original language and prepare question for processing
            original_question_lang = self.detect_language(question)
            processed_question = question
            
            # Translate question to English if needed for better FAISS search
            if original_question_lang != 'en':
                translated_question = self._translate_text(question, original_question_lang, 'en')
                if translated_question:
                    processed_question = translated_question

            # Step 2: Find best matches using FAISS
            top_k_matches = self.find_best_matches_faiss(processed_question)
            best_dataset_response = top_k_matches[0][0] if top_k_matches else None
            best_dataset_score = top_k_matches[0][1] if top_k_matches else 0.0            # Step 2.5: Fetch semantically relevant conversation history if user_id provided
            relevant_context = ""
            if user_id and self.use_vector_memory and self.vector_memory:
                try:
                    # Get semantically relevant conversation history using vector memory
                    relevant_context = self.get_conversation_context(
                        user_id=user_id,
                        query=processed_question,  # Use the processed question for semantic search
                        access_token=access_token
                    )
                    logger.info("Retrieved semantically relevant conversation context")
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logger.warning(f"Failed to get vector-based conversation context: {e}\n{error_details}")
                    # Disable vector memory for this session if it fails
                    self.use_vector_memory = False
                    logger.warning("Vector memory has been disabled for this session due to errors")
            
            # Use provided context if no relevant context found
            if not relevant_context and context:
                relevant_context = context
                
            # Step 2.6: Add in-memory chat history to context if available
            enhanced_context = relevant_context or ""
            if chat_history and len(chat_history) > 0:
                # Get the last 5 exchanges from chat history
                recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
                history_context = "\n\nRECENT CONVERSATION HISTORY (CRITICAL - YOU MUST REFERENCE THIS WHEN ANSWERING):\n"
                for i, exchange in enumerate(recent_history):
                    if len(exchange) >= 2:  # Ensure we have both question and answer
                        history_context += f"Exchange {i+1}:\nHuman: {exchange[0]}\nAssistant: {exchange[1]}\n\n"
                enhanced_context = history_context + ("\n" + enhanced_context if enhanced_context else "")
            # Step 2.7: Always include last 6 recency-based conversations for LLM context
            if user_id:
                try:
                    recent_convs = self.supabase.get_recent_conversations(
                        user_id=user_id,
                        access_token=access_token,
                        limit=6
                    )
                    recency_str = "\n\nPREVIOUS 6 CONVERSATIONS (RECENCY-BASED):\n"
                    for i, conv in enumerate(recent_convs):
                        recency_str += f"Exchange {i+1}: Human: {conv['message']}\nAssistant: {conv['response']}\n"
                    enhanced_context += recency_str
                except Exception as e:
                    logger.warning(f"Could not fetch recent conversations for context: {e}")

            # Debug logging to verify enhanced context construction
            logger.debug(f"Enhanced Context for LLM:\n{enhanced_context}")
            logger.debug(f"Chat History Length: {len(chat_history) if chat_history else 0}")
            logger.debug(f"Relevant Context Length: {len(relevant_context) if relevant_context else 0}")

            # Step 3: Determine response strategy based on confidence and language
            final_response_text = ""
            final_source = ""
            final_confidence = 0.0

            if best_dataset_response and best_dataset_score >= self.semantic_similarity_threshold:
                if original_question_lang == 'hi':
                    # For Hindi questions, do not enhance the dataset answer; return it as-is.
                    final_response_text = best_dataset_response
                    final_source = 'dataset (FAISS)'
                else:
                    # For non-Hindi questions, enhance the dataset answer using LLM.
                    enhanced_response = self.enhance_with_llm(question, best_dataset_response)
                    final_response_text = enhanced_response
                    final_source = 'dataset (FAISS, enhanced by LLM)'
                final_confidence = best_dataset_score
                
            elif self.use_llm_fallback:
                # Use LLM with context documents and conversation history
                context_documents = [match[0] for match in top_k_matches if match[0]]
                # Pass recency-enriched context into LLM
                llm_response = self.get_llm_response(
                    processed_question,
                    context_documents,
                    conversation_context=enhanced_context
                )
                
                if llm_response:
                    final_response_text = llm_response
                    final_source = 'llm (augmented with FAISS)' if context_documents else 'llm (no context)'
                    if enhanced_context:
                        final_source += ' with conversation context'
                    final_confidence = 75.0  # Fixed confidence for LLM responses
                else:
                    final_response_text = (best_dataset_response or "I'm sorry, I couldn't find a suitable response.")
                    final_source = 'dataset (low confidence)'
                    final_confidence = best_dataset_score
            else:
                final_response_text = (best_dataset_response or "I'm sorry, I couldn't find a suitable response.")
                final_source = 'dataset (low confidence)'
                final_confidence = best_dataset_score

            # Step 4: Add conversational discourse marker
            conversational_prefix = self._get_discourse_marker() + " "
            final_response_text = conversational_prefix + final_response_text

            # Step 5: Handle language translation and transliteration
            if original_question_lang != 'en':
                # Translate response back to original language
                translated_response = self._translate_text(final_response_text, 'en', original_question_lang)
                if translated_response:
                    final_response_text = translated_response
                    final_source = f"{final_source} (translated back to {original_question_lang})"

            # Step 6: Handle Hinglish transliteration if needed
            if original_question_lang == 'hi' and self._is_hinglish(question):
                hinglish_raw = transliterate(final_response_text, sanscript.DEVANAGARI, sanscript.ITRANS)
                final_response_text = self.polish_hinglish(hinglish_raw)
                final_source += " (transliterated and polished Hinglish)"

            # Step 7: Store conversation in vector memory for deep contextual recall
            if user_id and self.use_vector_memory and getattr(self, 'vector_memory', None):
                try:
                    self.vector_memory.store_memory(user_id, question, final_response_text)
                    logger.debug(f"Stored memory for user {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to store memory: {e}")
            return final_response_text.strip(), final_source, final_confidence
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            discourse_marker = self._get_discourse_marker()
            return (
                f"{discourse_marker} I apologize, but I encountered an error while processing your request. Please try again.",
                "error",
                0.0
            )