import json
import boto3
import pandas as pd
import numpy as np
import os
import faiss
from typing import Optional, Tuple, List, Dict
from langdetect import detect
from botocore.exceptions import ClientError
import random
from .utils import Config, load_qa_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION = "us-west-2"

class ResponseGeneratorRAG:
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

        self.bedrock_client = None
        self.embedding_client = None
        self.translate_client = None
        self.faiss_index = None
        self.qa_embeddings_array = None

        self._setup_aws_clients()
        self._load_faiss_artifacts(faiss_index_path, embeddings_file_path)

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
            print("Hinglish polish error:", e)
            return raw_text

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
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
        except:
            return text

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
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
        except:
            return None

    def find_best_matches_faiss(self, question: str) -> List[Tuple[str, float]]:
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
            similarity_score = 1 - (dist**2) / 2 
            confidence_score = max(0.0, min(100.0, similarity_score * 100))
            response_text = self.qa_dataset.iloc[idx]['Response']
            results.append((response_text, confidence_score))
        return results

    def get_llm_response(self, question: str, context_documents: Optional[List[str]] = None) -> Optional[str]:
        if not self.use_llm_fallback:
            return None
        try:
            context_str = ""
            if context_documents:
                context_str = "\n\nContext:\n" + "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_documents)]) + "\n"

            full_prompt = f"""You are a helpful, empathetic customer service assistant. Your primary goal is to answer the user's question in English (or in the original language if the question is not in English) in a sugar-coated, friendly, and empathetic tone. If the provided context documents are in a different language, translate them internally to English before answering. Answer the user's question ONLY using the provided context. If the context is insufficient, state that you cannot answer based on the provided information. (For example, instead of replying "bhai ek baat sun..." or "hum SBI ke employees hain..." in a harsh tone, sugar coat your reply so that it sounds empathetic and friendly.) In addition, please use respectful phrases (e.g. "bataiye", "boliye", "haan grahak ji") and avoid informal phrases (like "haa bhai"). {context_str}Question: {question}\n\nResponse:"""

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
        except:
            return None

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

    def get_response(self, question: str) -> Tuple[str, str, float]:
        original_question_lang = self.detect_language(question)
        processed_question = question
        if original_question_lang != 'en':
            translated_question = self._translate_text(question, original_question_lang, 'en')
            if translated_question:
                processed_question = translated_question

        top_k_matches = self.find_best_matches_faiss(processed_question)
        best_dataset_response = top_k_matches[0][0] if top_k_matches else None
        best_dataset_score = top_k_matches[0][1] if top_k_matches else 0.0

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
            context_documents = [match[0] for match in top_k_matches if match[0]]
            llm_response = self.get_llm_response(processed_question, context_documents)
            if llm_response:
                final_response_text = llm_response
                final_source = 'llm (augmented with FAISS)' if context_documents else 'llm (no context)'
                final_confidence = 100.0
            else:
                final_response_text = (best_dataset_response or "I'm sorry, I couldn't find a suitable response.")
                final_source = 'dataset (low confidence)'
                final_confidence = best_dataset_score
        else:
            final_response_text = (best_dataset_response or "I'm sorry, I couldn't find a suitable response.")
            final_source = 'dataset (low confidence)'
            final_confidence = best_dataset_score

        conversational_prefix = self._get_discourse_marker() + " "
        final_response_text = conversational_prefix + final_response_text

        if original_question_lang != 'en':
            translated_response = self._translate_text(final_response_text, 'en', original_question_lang)
            if translated_response:
                final_response_text = translated_response
                final_source = f"{final_source} (translated back to {original_question_lang})"

        if original_question_lang == 'hi' and self._is_hinglish(question):
            hinglish_raw = transliterate(final_response_text, sanscript.DEVANAGARI, sanscript.ITRANS)
            final_response_text = self.polish_hinglish(hinglish_raw)
            final_source += " (transliterated and polished Hinglish)"

        return final_response_text, final_source, final_confidence

    def get_response(self, question: str, context: Optional[str] = None) -> Tuple[str, str, float]:
        """Generate a response using both conversation context and dataset matching."""
        original_question_lang = self.detect_language(question)
        processed_question = question

        if original_question_lang != 'en':
            translated_question = self._translate_text(question, original_question_lang, 'en')
            if translated_question:
                processed_question = translated_question

        # Find matches from the dataset
        top_k_matches = self.find_best_matches_faiss(processed_question)
        best_dataset_response = top_k_matches[0][0] if top_k_matches else None
        best_dataset_score = top_k_matches[0][1] if top_k_matches else 0.0

        if best_dataset_response and best_dataset_score >= self.semantic_similarity_threshold:
            if original_question_lang == 'hi':
                # For Hindi questions, return dataset answer as-is
                final_response_text = best_dataset_response
                final_source = 'dataset (FAISS)'
            else:
                # For non-Hindi questions, enhance with context
                context_enhanced_response = self.enhance_with_context(
                    question=processed_question,
                    dataset_answer=best_dataset_response,
                    context=context
                )
                final_response_text = context_enhanced_response
                final_source = 'dataset (FAISS + context)'
            final_confidence = best_dataset_score
        elif self.use_llm_fallback:
            # Use conversation context and any relevant dataset matches
            context_documents = [match[0] for match in top_k_matches if match[0]]
            if context:
                context_documents.insert(0, context)
            
            llm_response = self.get_llm_response(processed_question, context_documents)
            if llm_response:
                final_response_text = llm_response
                final_source = 'llm (with context)' if context else 'llm (no context)'
                final_confidence = 100.0
            else:
                final_response_text = (best_dataset_response or "I apologize, but I couldn't find a suitable response.")
                final_source = 'dataset (low confidence)'
                final_confidence = best_dataset_score
        else:
            final_response_text = (best_dataset_response or "I apologize, but I couldn't find a suitable response.")
            final_source = 'dataset (low confidence)'
            final_confidence = best_dataset_score

        # Add conversation markers and handle translation
        conversational_prefix = self._get_discourse_marker() + " "
        final_response_text = conversational_prefix + final_response_text

        if original_question_lang != 'en':
            translated_response = self._translate_text(final_response_text, 'en', original_question_lang)
            if translated_response:
                final_response_text = translated_response
                final_source = f"{final_source} (translated to {original_question_lang})"

        if original_question_lang == 'hi' and self._is_hinglish(question):
            hinglish_raw = transliterate(final_response_text, sanscript.DEVANAGARI, sanscript.ITRANS)
            final_response_text = self.polish_hinglish(hinglish_raw)
            final_source += " (transliterated and polished Hinglish)"

        return final_response_text, final_source, final_confidence

    def enhance_with_context(
        self,
        question: str,
        dataset_answer: str,
        context: Optional[str] = None
    ) -> str:
        """Enhance the response using conversation context and LLM."""
        if not self.bedrock_client or not context:
            return dataset_answer

        try:
            prompt = f"""You are a helpful, empathetic customer service assistant. Please provide a response to the user's question that takes into account both the suggested answer and the conversation history. Make the response personal and contextually relevant while maintaining a professional tone.

Previous Conversation:
{context}

Current Question: {question}
Suggested Answer: {dataset_answer}

Please provide a contextually enhanced response (limit to 100 words):"""

            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.4,
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
            print(f"Warning: Could not enhance response with context: {e}")
            return dataset_answer