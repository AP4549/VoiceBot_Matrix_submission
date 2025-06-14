import json
import boto3
import pandas as pd
import numpy as np
import os
import faiss
from typing import Optional, Tuple, List
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
    _discourse_markers = {
        'en': [
            "Hmm, let me see...",
            "Okay, got it.",
            "Just a moment...",
            "Thinking...",
            "Right, let's find that out."
        ],
        'hi': [
            "ठीक है, देखता हूं...",
            "एक मिनट...",
            "सोच रहा हूं...",
            "हां, बताता हूं...",
            "समझ गया..."
        ]
    }
    
    _fallback_responses = {
        'en': [
            "I'd like to help you. Could you please provide more details?",
            "I'm not quite sure I understood. Could you rephrase that?",
            "Could you elaborate on your question?"
        ],
        'hi': [
            "मैं आपकी मदद करना चाहूंगा। कृपया अधिक जानकारी दें।",
            "क्षमा करें, मैं आपका प्रश्न पूरी तरह नहीं समझ पाया। क्या आप इसे दोबारा बता सकते हैं?",
            "क्या आप अपने प्रश्न को और विस्तार से बता सकते हैं?"
        ]
    }

    def __init__(self, faiss_index_path: str, embeddings_file_path: str):
        self.config = Config()
        self.qa_dataset = load_qa_dataset()
        self.semantic_similarity_threshold = self.config.get('response', {}).get('semantic_similarity_threshold', 0.50)
        self.use_llm_fallback = self.config.get('response', {}).get('use_llm_fallback', True)
        self.top_k_retrieval = self.config.get('rag', {}).get('top_k_retrieval', 3)

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

    def _get_discourse_marker(self, language: str = 'en') -> str:
        markers = self._discourse_markers.get(language, self._discourse_markers['en'])
        return random.choice(markers)

    def detect_language(self, text: str) -> str:
        """Improved language detection with better Hindi recognition"""
        try:
            # Check for Devanagari characters first
            devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
            if devanagari_count > 0:
                return 'hi'

            # If no Devanagari, check for Hinglish
            if self._is_hinglish(text):
                return 'hi'

            # Use langdetect as fallback
            lang = detect(text)
            if lang not in ['en', 'hi']:
                # Check if the text contains common Hindi/Hinglish words
                common_hindi_words = ['hai', 'kya', 'main', 'nahi', 'aap', 'kaise', 'mujhe', 'hum', 'tum']
                text_lower = text.lower()
                if any(word in text_lower.split() for word in common_hindi_words):
                    return 'hi'
            return lang if lang in ['en', 'hi'] else 'en'
        except:
            return 'en'  # Default to English on error

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
                context_str = "\n".join([f"- {doc}" for doc in context_documents])

            # Detect input format
            input_lang = self.detect_language(question)
            is_hinglish = self._is_hinglish(question)

            # Determine required response format and script
            response_format = "Hinglish" if is_hinglish else ("Hindi" if input_lang == 'hi' else "English")
            script_example = "roman script" if is_hinglish else ("देवनागरी script" if input_lang == 'hi' else "English")

            language_instruction = ""
            if response_format == "Hindi":
                language_instruction = """CRITICAL: 
- Respond COMPLETELY in Hindi using Devanagari script only
- Use proper banking terminology in Hindi
- Format lists and steps clearly with numbers or bullet points
- DO NOT use ANY English words"""
            elif response_format == "Hinglish":
                language_instruction = """CRITICAL:
- Respond in natural Hinglish (Hindi words in Roman script)
- Match the casual tone and script style of the question
- Use common Hinglish banking terms that people understand
- Keep the response conversational"""
            else:  # English
                language_instruction = """CRITICAL:
- Respond in clear, simple English
- Use standard banking terminology
- Format information in clear, numbered lists when needed
- Keep the tone professional but friendly"""

            full_prompt = f"""You are an Indian bank's AI P2P assistant.

USER'S QUESTION (in {response_format}):
{question}

CONTEXT INFORMATION:
{context_str}

RESPONSE REQUIREMENTS:
{language_instruction}

FORMATTING RULES:
1. Keep response under 100 words
2. Use bullet points or numbers for lists
3. Structure information clearly
4. Include specific details from context
5. Be warm and helpful

YOUR RESPONSE IN {response_format}:"""

            messages = [{"role": "user", "content": full_prompt}]
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.3,
                    "top_p": 0.9,
                })
            )
            response_body = json.loads(response['body'].read())
            llm_generated_text = ''
            if response_body.get('content'):
                for content_block in response_body['content']:
                    if content_block.get('type') == 'text':
                        llm_generated_text = content_block['text'].strip()
            
            # Additional language enforcement check
            if input_lang == 'hi' and not is_hinglish and not any('\u0900' <= c <= '\u097F' for c in llm_generated_text):
                # Force convert to Hindi if response came in English
                llm_generated_text = self._ensure_language_match(llm_generated_text, 'hi')
            
            return llm_generated_text

        except Exception as e:
            print(f"Error in LLM response generation: {e}")
            return None

    def get_response(self, question: str) -> Tuple[str, str, float]:
        # Detect input format
        original_question_lang = self.detect_language(question)
        is_hinglish = self._is_hinglish(question)
        
        # Process question for FAISS search (always in English for better matching)
        processed_question = question
        if original_question_lang == 'hi':
            translated_question = self._translate_text(question, 'hi', 'en')
            if translated_question:
                processed_question = translated_question

        # Get response using FAISS
        top_k_matches = self.find_best_matches_faiss(processed_question)
        best_dataset_response = top_k_matches[0][0] if top_k_matches else None
        best_dataset_score = top_k_matches[0][1] if top_k_matches else 0.0

        # Get response based on confidence
        if best_dataset_response and best_dataset_score >= self.semantic_similarity_threshold:
            final_response_text = best_dataset_response
            final_source = 'dataset (FAISS)'
            final_confidence = best_dataset_score
        elif self.use_llm_fallback:
            context_documents = [match[0] for match in top_k_matches if match[0]]
            llm_response = self.get_llm_response(question, context_documents)  # Pass original question for language context
            if llm_response:
                final_response_text = llm_response
                final_source = 'llm (augmented with FAISS)' if context_documents else 'llm (no context)'
                final_confidence = 100.0
            else:
                final_response_text = best_dataset_response or "I'm sorry, I couldn't find a suitable response."
                final_source = 'dataset (low confidence)'
                final_confidence = best_dataset_score
        else:
            final_response_text = best_dataset_response or "I'm sorry, I couldn't find a suitable response."
            final_source = 'dataset (low confidence)'
            final_confidence = best_dataset_score

        # Add discourse marker in appropriate language
        conversational_prefix = self._get_discourse_marker(original_question_lang) + " "
        final_response_text = conversational_prefix + final_response_text

        # Handle response language format based on input format
        if original_question_lang == 'hi':
            if is_hinglish:
                # For Hinglish questions, ensure Hinglish response
                if not self._is_hinglish(final_response_text):
                    hinglish_raw = transliterate(final_response_text, sanscript.DEVANAGARI, sanscript.ITRANS)
                    final_response_text = self.polish_hinglish(hinglish_raw)
                    final_source += " (converted to Hinglish)"
            else:
                # For pure Hindi (Devanagari) questions, ensure Hindi response
                if not any('\u0900' <= c <= '\u097F' for c in final_response_text):
                    translated_response = self._translate_text(final_response_text, 'en', 'hi')
                    if translated_response:
                        final_response_text = translated_response
                        final_source += " (translated to Hindi)"
        elif original_question_lang == 'en':
            # For English questions, ensure English response
            if any('\u0900' <= c <= '\u097F' for c in final_response_text):
                translated_response = self._translate_text(final_response_text, 'hi', 'en')
                if translated_response:
                    final_response_text = translated_response
                    final_source += " (translated to English)"

        return final_response_text, final_source, final_confidence

    def _get_language_specific_response(self, query: str, detected_lang: str, context: str = None) -> str:
        """Generate a response in the same language as the query using RAG and LLM"""
        if not context:
            return random.choice(self._fallback_responses[detected_lang])

        # Create a detailed prompt for context-aware response
        prompt = f"""You are a helpful banking assistant at a major Indian bank. Generate a natural response based on the following rules:

Question: {query}
Context: {context}

Response Rules:
1. Language:
   - For Hindi questions (in Devanagari): Respond COMPLETELY in Hindi using Devanagari script
   - For English questions: Respond in English
   - For Hinglish: Match the language mix of the question

2. Content:
   - Use the provided context to create an accurate response
   - Include specific steps, requirements, or processes when applicable
   - Add relevant details about documentation or eligibility if mentioned
   - For product queries, mention key features and benefits

3. Style:
   - Be professional yet conversational
   - Keep responses concise but informative
   - Use bullet points or numbered lists for steps
   - Be polite and helpful

4. Banking Terms:
   - Use proper banking terminology in the respective language
   - Explain technical terms if needed
   - Include relevant financial information when available

Response:"""
            
        try:            
            # Request a concise response by adding to the prompt
            prompt += "\nIMPORTANT: Keep your response under 50 words and focus on the most important points."                # Add strict language requirement to prompt
                
            if detected_lang == 'hi':
                    prompt += "\nCRITICAL: You MUST respond ONLY in Hindi using Devanagari script. DO NOT use ANY English words."
            else:
                    prompt += "\nCRITICAL: You MUST respond ONLY in English. DO NOT mix languages."

            response = self.bedrock_client.invoke_model(
                    modelId=CLAUDE_MODEL_ID,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.3,
                    })
            )            
            content_blocks = json.loads(response["body"].read()).get("content", [])
            if content_blocks and content_blocks[0].get("type") == "text":
                response_text = content_blocks[0]["text"].strip()
                # Ensure response is within 50 words
                return self._limit_response_length(response_text, 50)
        except Exception as e:
            print(f"Error generating language-specific response: {e}")
        
        # Fallback to pre-defined templates if no context or error
        if detected_lang == 'hi':
            return "मुझे खुशी होगी आपकी मदद करने में। कृपया अपना प्रश्न और विस्तार से बताएं।"
        return "I apologize, but I couldn't find a specific answer. Could you please rephrase your question?"

    def generate_response(self, query: str, detected_lang: str = None) -> Tuple[str, float, str]:
        """Generate a response using RAG with dynamic language handling"""
        if not detected_lang:
            detected_lang = self.detect_language(query)

        # Add a discourse marker
        response = random.choice(self._discourse_markers[detected_lang]) + "\n\n"
        
        try:
            # Get query embedding and search similar content
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return response + random.choice(self._fallback_responses[detected_lang]), 0.0, detected_lang

            # Search FAISS index for relevant content
            D, I = self.faiss_index.search(np.array([query_embedding]), self.top_k_retrieval)
            max_similarity = float(D[0][0]) if len(D) > 0 and len(D[0]) > 0 else 0.0
            
            # If we have good matches, use them for context
            if max_similarity >= self.semantic_similarity_threshold:
                contexts = []
                for idx in I[0]:
                    if idx < len(self.qa_dataset):
                        qa_pair = self.qa_dataset.iloc[idx]
                        contexts.append(f"Q: {qa_pair['Question']}\nA: {qa_pair['Response']}")
                  # Create prompt with context
                context_text = "\n".join(contexts)
                language_instruction = (
                    "IMPORTANT: Your response MUST be completely in Hindi using Devanagari script only. DO NOT use any English words."
                    if detected_lang == 'hi' else
                    "Respond in clear, professional English."
                )
                
                prompt = (
                    f"You are a helpful Indian banking assistant. {language_instruction}\n\n"
                    f"User Question: {query}\n\n"
                    f"Relevant Context:\n{context_text}\n\n"
                    "Response Requirements:\n"
                    f"1. Language: {language_instruction}\n"
                    "2. Banking Terms: Use appropriate terminology in the required language\n"
                    "3. Format: Use clear numbering for steps or requirements\n\n"
                    "2. Content Guidelines:\n"
                    "   - Use the provided context intelligently\n"
                    "   - Be specific about processes, documents, and requirements\n"
                    "   - Include relevant banking terms in the appropriate language\n"
                    "   - Keep it concise but informative\n\n"
                    "3. Style:\n"
                    "   - Be professional yet conversational\n"
                    "   - Use clear formatting (bullets/numbers for steps)\n"
                    "   - Maintain proper banking terminology\n\n"
                    "Response:"
                )
                try:
                    response_text = self._generate_llm_response(prompt)
                    if response_text:
                        # Enforce the response language to match the input
                        enforced_response = self._enforce_language(response_text, detected_lang)
                        return response + enforced_response, max_similarity * 100, detected_lang
                except Exception as e:
                    print(f"Error in LLM response generation: {e}")
            
            # Fallback to simple response
            return response + random.choice(self._fallback_responses[detected_lang]), max_similarity * 100, detected_lang
            
        except Exception as e:
            print(f"Error in response generation: {e}")
            return response + random.choice(self._fallback_responses[detected_lang]), 0.0, detected_lang

    def _generate_llm_response(self, prompt: str) -> Optional[str]:
        """Generate response using Claude model"""
        try:
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                })
            )
            content_blocks = json.loads(response["body"].read()).get("content", [])
            return content_blocks[0]["text"].strip() if content_blocks and content_blocks[0].get("type") == "text" else None
        except Exception as e:
            print(f"LLM generation error: {e}")
            return None

    def _enforce_language(self, text: str, target_lang: str) -> str:
        """Ensure the response is in the correct language"""
        if not text:
            return ""
            
        # Check if the text is already in the correct language
        detected = self._detect_language(text)
        if detected == target_lang:
            return text
            
        # If not, create a translation prompt
        if target_lang == 'hi':
            prompt = f"""Translate this banking response to natural, conversational Hindi using Devanagari script. 
Keep banking terms clear and understandable. Maintain any lists or step-by-step instructions:

Text: {text}

Hindi Translation (in Devanagari):"""
        else:
            prompt = f"""Translate this banking response to natural, professional English.
Keep banking terms clear and understandable. Maintain any lists or step-by-step instructions:

Text: {text}

English Translation:"""
            
        try:
            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                })
            )
            content_blocks = json.loads(response["body"].read()).get("content", [])
            translated = content_blocks[0]["text"].strip() if content_blocks and content_blocks[0].get("type") == "text" else text
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def _limit_response_length(self, text: str, max_words: int = 50) -> str:
        """Limit response to specified number of words while maintaining sentence completion"""
        if not text:
            return text
            
        words = text.split()
        if len(words) <= max_words:
            return text
            
        # Try to find a sentence end within reasonable range of max_words
        shortened = " ".join(words[:max_words])
        last_period = shortened.rfind("।") if "।" in shortened else shortened.rfind(".")
        
        if last_period != -1:
            return shortened[:last_period + 1]
        
        # If no sentence end found, just cut at word boundary
        return " ".join(words[:max_words]) + "..."
    
    def _ensure_language_match(self, text: str, target_lang: str) -> str:
        """Ensure the response is in the same language as the input"""
        try:
            # Check if text contains Devanagari for Hindi
            has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
            
            if target_lang == 'hi' and not has_devanagari:
                # Convert English response to Hindi
                prompt = f"""Translate this banking response to natural Hindi using Devanagari script only. 
Keep banking terms clear and easy to understand.

Text: {text}

Hindi Translation:"""
            elif target_lang == 'en' and has_devanagari:
                # Convert Hindi response to English
                prompt = f"""Translate this banking response to natural English.
Keep banking terms clear and professional.

Text: {text}

English Translation:"""
            else:
                return text  # Already in correct language

            response = self.bedrock_client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3
                })
            )
            
            content = json.loads(response["body"].read()).get("content", [])
            if content and content[0].get("type") == "text":
                return content[0]["text"].strip()
            
            return text
        except Exception as e:
            print(f"Language conversion error: {e}")
            return text