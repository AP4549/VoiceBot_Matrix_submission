import os
import gradio as gr
import tempfile
import json
import pandas as pd
import datetime
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from modules.aws_asr import AWSTranscribeModule
from modules.response_gen_rag import ResponseGeneratorRAG
from modules.response_gen_ragb import ResponseGeneratorRAGB
from modules.aws_tts import AWSTTSModule
from modules.supabase_client import SupabaseManager
import boto3
from modules.utils import Config, get_aws_credentials, clean_text
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for better Gradio stability
os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
os.environ["GRADIO_SERVER_PORT"] = "7860"

def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry function calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final attempt failed for {func.__name__}: {e}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

class VoiceBotUI:
    def __init__(self):
        """Initialize VoiceBot with proper error handling"""
        try:
            logger.info("Initializing VoiceBot components...")
            self.config = Config()
            
            # Initialize user state
            self.current_user = None
            self.access_token = None
            
            # Initialize chat history
            self.chat_history = []
            
            # Initialize Supabase
            try:
                self.supabase = SupabaseManager()
                logger.info("✅ Supabase client initialized")
            except Exception as supabase_error:
                logger.error(f"❌ Supabase initialization failed: {supabase_error}")
                raise
            
            # Initialize modules with error handling
            self._initialize_modules()
            
            # Setup directories for audio files
            self._setup_directories()
            
            logger.info("✅ VoiceBot initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VoiceBot: {e}")
            raise

    def sign_in_user(self, email: str, password: str) -> Tuple[bool, str]:
        """Sign in a user and store their session"""
        try:
            response = self.supabase.sign_in(email, password)
            if response and hasattr(response, 'session') and response.session:
                self.current_user = response.user
                self.access_token = response.session.access_token
                return True, "Successfully logged in!"
            return False, "Login failed - invalid credentials"
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return False, f"Login failed: {str(e)}"

    def sign_out_user(self) -> Tuple[bool, str]:
        """Sign out the current user"""
        try:
            if self.current_user:
                self.supabase.sign_out()
                self.current_user = None
                self.access_token = None
                return True, "Successfully logged out!"
            return False, "No user is currently logged in"
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False, f"Logout failed: {str(e)}"

    def _initialize_modules(self):
        """Initialize ASR, TTS, and RAG modules with error handling"""
        try:
            # Initialize ASR module
            self.asr_module = AWSTranscribeModule()
            logger.info("✅ ASR module initialized")
            
            # Initialize TTS module
            self.tts_module = AWSTTSModule()
            logger.info("✅ TTS module initialized")
            
            # Define paths for FAISS artifacts
            faiss_index_path = "Data/qa_faiss_index.bin"
            embeddings_file_path = "Data/qa_embeddings.npy"
            
            # Check if FAISS files exist
            if not os.path.exists(faiss_index_path):
                logger.warning(f"⚠️ FAISS index not found: {faiss_index_path}")
            if not os.path.exists(embeddings_file_path):
                logger.warning(f"⚠️ Embeddings file not found: {embeddings_file_path}")
            
            # Initialize both RAG implementations
            self.response_gen = ResponseGeneratorRAG(
                faiss_index_path=faiss_index_path, 
                embeddings_file_path=embeddings_file_path
            )
            self.response_gen_with_context = ResponseGeneratorRAGB(
                faiss_index_path=faiss_index_path, 
                embeddings_file_path=embeddings_file_path
            )
            logger.info("✅ RAG response generators initialized")
            
        except Exception as e:
            logger.error(f"❌ Module initialization failed: {e}")
            raise

    def _setup_directories(self):
        """Create necessary directories for audio files"""
        try:
            self.input_dir = Path("Data/userinputvoice")
            self.output_dir = Path("Data/voicebotoutput")
            self.audio_dir = Path("Data/userinputvoice/audio")
            
            # Create directories
            self.input_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ Directories setup complete")
            
        except Exception as e:
            logger.error(f"❌ Directory setup failed: {e}")
            raise

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text with improved accuracy"""
        try:
            if not text or not text.strip():
                return 'en'
            
            # Check for Devanagari characters first
            devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
            if devanagari_count > 0:
                return 'hi'
            
            # Use langdetect for other cases
            lang = detect(text)
            if lang not in ['en', 'hi']:
                # Check for common Hindi/Hinglish words
                hindi_words = ['hai', 'kya', 'main', 'nahi', 'aap', 'kaise', 'mujhe', 'hum', 'tum', 'kar', 'koi']
                text_lower = text.lower()
                if any(word in text_lower.split() for word in hindi_words):
                    return 'hi'
                return 'en'  # Default to English
            return lang
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default to English on error

    def _is_hinglish(self, text: str) -> bool:
        """Check if the text is Hinglish (Hindi written in Roman script)"""
        if not text:
            return False
        try:
            devanagari_count = sum('\u0900' <= c <= '\u097F' for c in text)
            latin_count = sum('a' <= c.lower() <= 'z' for c in text if c.isalpha())
            return devanagari_count == 0 and latin_count > 3
        except:
            return False

    @retry_on_failure(max_retries=2)
    def _save_audio(self, audio_path, audio_data, sample_rate):
        """Save audio data to file with retry logic"""
        try:
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure proper format
            audio_data = np.float32(audio_data)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(str(audio_path)), exist_ok=True)
            
            # Save audio file
            sf.write(str(audio_path), audio_data, sample_rate)
            logger.info(f"✅ Audio saved successfully: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio saving failed: {e}")
            raise

    @retry_on_failure(max_retries=2)
    def speech_to_text(self, audio_path):
        """Convert speech to text using AWS Transcribe through ASR module"""
        try:
            logger.info(f"Converting speech to text from: {audio_path}")
            transcript = self.asr_module.transcribe_audio(audio_path)
            if transcript:
                logger.info(f"Transcription successful: {transcript[:50]}...")
                return transcript
            else:
                logger.warning("Transcription failed or returned empty result")
                return None
        except Exception as e:
            logger.error(f"❌ Speech-to-text conversion error: {e}")
            return None    
    def process_audio(self, audio_path):
        """Process audio input and return (transcript, response, audio_response, history)"""
        try:
            # Get transcript
            transcript = self.speech_to_text(audio_path) or ""
            if not transcript:
                # No valid transcript
                return "", "I couldn't understand the audio. Could you please try again?", None, []

            clean_transcript = transcript.strip()
            detected_lang = detect(clean_transcript) if clean_transcript else 'en'
            
            # Get conversation context from Supabase for authenticated users
            context = self.get_conversation_context()
            
            # Log chat history before generating response
            logger.info(f"Chat history before response generation: {self.chat_history}")
            logger.info(f"Chat history length: {len(self.chat_history)}")
            
            # User ID for vector memory retrieval
            user_id = self.current_user.id if self.current_user else None
            
            # Generate response with context, chat history, and user ID for vector memory
            response, source, confidence = self.response_gen_with_context.get_response(
                clean_transcript,
                context=context,
                chat_history=self.chat_history,
                user_id=user_id,
                access_token=self.access_token
            )
            
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                source = "fallback"
                confidence = 0.0
            
            # Generate audio response
            audio_response = self.text_to_speech(response, detected_lang)
            # Ensure audio_response is a valid filepath or None (file must exist)
            if not audio_response or not os.path.isfile(audio_response):
                audio_response = None
            
            # Store conversation if authenticated
            if self.current_user and self.access_token:
                try:
                    # Store conversation in Supabase
                    self.supabase.store_conversation(
                        user_id=self.current_user.id,
                        message=clean_transcript,
                        response=response,
                        audio_url=audio_path,
                        response_audio_url=audio_response,
                        language=detected_lang,
                        confidence_score=confidence,
                        source=source,
                        access_token=self.access_token
                    )
                    
                    # Store in vector memory if available
                    if hasattr(self.response_gen_with_context, 'vector_memory') and self.response_gen_with_context.vector_memory:
                        # Get embedding for the combined text
                        combined_text = f"Human: {clean_transcript}\nAssistant: {response}"
                        
                        try:
                            # Store in vector memory for semantic retrieval
                            success = self.response_gen_with_context.vector_memory.store_memory(
                                user_id=self.current_user.id,
                                message=clean_transcript,
                                response=response,
                                importance=1.0,  # Default importance
                                metadata={
                                    "language": detected_lang,
                                    "confidence": confidence,
                                    "source": source
                                }
                            )
                            if success:
                                logger.info("Successfully stored conversation in vector memory")
                            else:
                                logger.warning("Failed to store conversation in vector memory")
                        except Exception as vm_error:
                            logger.warning(f"Vector memory storage error: {vm_error}")
                    
                except Exception as e:
                    logger.warning(f"Failed to store conversation: {str(e)}")
            
            # Update in-memory chat history
            self.chat_history.append([clean_transcript, response])
            
            # Log chat history after updating
            logger.info(f"Chat history after update: {self.chat_history}")
            logger.info(f"Updated chat history length: {len(self.chat_history)}")
            
            # Return results
            return transcript, response, audio_response, self.chat_history
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            error_msg = f"Processing error: {str(e)}"
            # Return safe defaults
            return "", "An error occurred while processing your request.", None, []

    @retry_on_failure(max_retries=2)
    def text_to_speech(self, text: str, language: str = 'en') -> Optional[str]:
        """Convert text to speech using AWS Polly through TTS module"""
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for TTS")
            return None
        
        try:
            logger.info(f"Converting text to speech (language: {language})")
            audio_path = self.tts_module.synthesize_speech(text, language)
            if audio_path:
                logger.info("✅ TTS conversion successful")
            else:
                logger.warning("⚠️ TTS conversion returned None")
            return audio_path
        except Exception as e:
            logger.error(f"❌ TTS conversion failed: {e}")
            return None

    def get_conversation_context(self) -> str:
        """Get conversation context from Supabase for the current user."""
        if self.current_user and self.access_token:
            try:
                context = self.response_gen_with_context.get_conversation_context(
                    user_id=self.current_user.id,
                    access_token=self.access_token
                )
                return context
            except Exception as e:
                logger.warning(f"Failed to get Supabase conversation context: {str(e)}")
        return ""

    def process_text_input(self, text_input: str, chat_history: List[List[str]]) -> str:
        """Process text input from the chat interface and return (response)"""
        try:
            # Clean input and detect language
            clean_text_input = text_input.strip()
            detected_lang = detect(clean_text_input) if clean_text_input else 'en'
            
            # Fallback: if user asks about previous question/topic
            lower = clean_text_input.lower()
            if any(kw in lower for kw in ["previous question", "my previous question", "what was my previous question", "what topic"]):
                if self.chat_history:
                    last_q = self.chat_history[-1][0]
                    return f"Your previous question was: '{last_q}'"
                else:
                    return "I don't have any previous question recorded."
            
            # Get conversation context from Supabase
            context = self.get_conversation_context()
            
            # Generate response with context and chat history
            response, source, confidence = self.response_gen_with_context.get_response(
                clean_text_input,
                context=context,
                chat_history=chat_history
            )
            
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                source = "fallback"
                confidence = 0.0
            
            # Store conversation if authenticated
            if self.current_user and self.access_token:
                try:
                    self.supabase.store_conversation(
                        user_id=self.current_user.id,
                        message=clean_text_input,
                        response=response,
                        language=detected_lang,
                        confidence_score=confidence,
                        source=source,
                        access_token=self.access_token
                    )
                except Exception as e:
                    logger.warning(f"Failed to store conversation: {str(e)}")
            
            # Update in-memory chat history
            self.chat_history.append([clean_text_input, response])
            
            return response
        except Exception as e:
            logger.error(f"❌ Text processing error: {e}")
            return f"Processing error: {str(e)}"

    def check_chat_history(self):
        """Check and log the current chat history"""
        try:
            history_length = len(self.chat_history)
            logger.info(f"Current chat history length: {history_length}")
            logger.info(f"Chat history contents: {self.chat_history}")
            
            history_summary = ""
            if history_length == 0:
                history_summary = "Chat history is empty. No previous conversations found."
            else:
                history_summary = f"Chat history contains {history_length} exchanges:\n"
                for i, exchange in enumerate(self.chat_history):
                    if len(exchange) >= 2:
                        history_summary += f"Exchange {i+1}:\nUser: {exchange[0]}\nAssistant: {exchange[1]}\n\n"
            
            return history_summary
        except Exception as e:
            logger.error(f"Failed to check chat history: {e}")
            return f"Error checking chat history: {str(e)}"

    def reset_chat_history(self):
        """Reset the in-memory chat history"""
        try:
            previous_length = len(self.chat_history)
            self.chat_history = []
            logger.info(f"Chat history reset. Previous length: {previous_length}")
            return True, "Chat history has been reset."
        except Exception as e:
            logger.error(f"Failed to reset chat history: {e}")
            return False, f"Failed to reset chat history: {str(e)}"

def create_gradio_interface(voicebot):
    """Create and configure Gradio interface"""
    try:
        # Custom CSS
        css = """
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .tab-nav {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        .small-btn button {
            width: 34px;
            height: 34px;
            padding: 0;
            font-size: 1rem;
            margin-right: 4px;
            border-radius: 4px;
        }
        """
        # Define enhanced theme
        theme = gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="blue",
            neutral_hue="slate",
            spacing_size="lg",
            radius_size="md"
        ).set(
            body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            panel_background_fill="rgba(255, 255, 255, 0.95)"
        )
        # Create blocks interface with custom CSS
        with gr.Blocks(title="Voice Assistant", theme=theme, css=css) as interface:
            gr.Markdown("# 🎙️ Voice Assistant")
            
            # Authentication section
            with gr.Tab("Authentication"):
                gr.Markdown("## 🔐 User Authentication")
                with gr.Row():
                    email_input = gr.Textbox(label="Email")
                    password_input = gr.Textbox(label="Password", type="password")
                with gr.Row():
                    login_button = gr.Button("Sign In")
                    logout_button = gr.Button("Sign Out")
                auth_status = gr.Textbox(label="Authentication Status", interactive=False)
                # Hidden flag to track login success
                auth_flag = gr.Textbox(visible=False)
            
            # Voice Assistant tab, hidden until login
            with gr.Tab("Voice Assistant", visible=False) as voice_tab:
                gr.Markdown("## 🗣️ Voice Interaction")
                with gr.Row():
                    with gr.Column(scale=1):
                        # Audio input and settings
                        audio_input = gr.Audio(type="filepath", label="Speak or Upload",
                                               sources=["microphone", "upload"])
                        file_upload = gr.File(label="Or upload audio file", file_types=[".wav", ".mp3"])
                        # Actions
                        process_btn = gr.Button("Process Audio")
                        # Reset and Check buttons removed, to be added as icons near chat history
                    with gr.Column(scale=2):
                        transcript_output = gr.Textbox(label="Transcript")
                        response_output = gr.Textbox(label="Response")
                        audio_output = gr.Audio(label="Voice Response")
                        chat_history_output = gr.Dataframe(
                             headers=["User", "Assistant"],
                             datatype=["str", "str"],
                             label="Chat History"
                         )
                        with gr.Row():
                            reset_btn_small = gr.Button("↺", elem_classes="small-btn")
                            history_btn_small = gr.Button("📜", elem_classes="small-btn")
                # Connect voice processing callbacks with built-in progress indicator
                process_btn.click(
                    fn=voicebot.process_audio,
                    inputs=[audio_input],
                    outputs=[transcript_output, response_output, audio_output, chat_history_output],
                    show_progress=True
                )
                file_upload.change(
                    fn=voicebot.process_audio,
                    inputs=[file_upload],
                    outputs=[transcript_output, response_output, audio_output, chat_history_output],
                    show_progress=True
                )
                # Connect minimal buttons
                reset_btn_small.click(fn=voicebot.reset_chat_history, inputs=[], outputs=[auth_status])
                history_btn_small.click(fn=voicebot.check_chat_history, inputs=[], outputs=[auth_status])
        
            # Chat Inference tab, hidden until login
            with gr.Tab("Chat Inference", visible=False) as chat_tab:
                gr.Markdown("## 💬 Text Chat Inference")
                chatbot = gr.Chatbot(label="Conversation")
                msg = gr.Textbox(show_label=False, placeholder="Type your message here and hit Send")
                send_btn = gr.Button("Send")
                # Define respond function to update chatbot
                def respond_text(message, history):
                    if not message:
                        return history, ""
                    # Use text processing to get response
                    response = voicebot.process_text_input(message, history)
                    history = history + [[message, response]]
                    return history, ""
                # Wire up send button
                send_btn.click(
                    fn=respond_text,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg]
                )
            
            # Connect authentication buttons
            login_button.click(
                fn=voicebot.sign_in_user,
                inputs=[email_input, password_input],
                outputs=[auth_status]
            )
            
            logout_button.click(
                fn=voicebot.sign_out_user,
                inputs=[],
                outputs=[auth_status]
            )
            
            # Hook login/logout to auth_flag and tab visibility
            login_button.click(fn=voicebot.sign_in_user, inputs=[email_input, password_input], outputs=[auth_flag, auth_status])
            login_button.click(
                fn=lambda flag: (gr.update(visible=flag), gr.update(visible=flag)),
                inputs=[auth_flag], outputs=[voice_tab, chat_tab]
            )
            logout_button.click(fn=voicebot.sign_out_user, inputs=[], outputs=[auth_status])
            logout_button.click(
                fn=lambda _: (gr.update(visible=False), gr.update(visible=False)),
                inputs=[auth_status], outputs=[voice_tab, chat_tab]
            )
            
            # Connect reset history button
            reset_btn_small.click(
                fn=voicebot.reset_chat_history,
                inputs=[],
                outputs=[auth_status]
            )
            
            # Connect check history button
            history_btn_small.click(
                fn=voicebot.check_chat_history,
                inputs=[],
                outputs=[auth_status]
            )
            
            # Auto-process when audio is recorded
            audio_input.stop_recording(
                fn=voicebot.process_audio,
                inputs=[audio_input],
                outputs=[transcript_output, response_output, audio_output, chat_history_output]
            )
        
            # About section
            with gr.Tab("About"):
                gr.Markdown("""
                ## About Voice Assistant
                
                This voice assistant uses AWS services for speech recognition and synthesis, 
                combined with a RAG (Retrieval Augmented Generation) system for intelligent responses.
                
                ### Features:
                - Speech-to-text using AWS Transcribe
                - Text-to-speech using AWS Polly
                - Intelligent responses using RAG and LLM
                - Multi-language support
                - User authentication and conversation history
                """)
        
        return interface
        
    except Exception as e:
        logger.error(f"Failed to create Gradio interface: {e}")
        raise

def main():
    """Main function with comprehensive error handling"""
    try:
        logger.info("🚀 Starting VoiceBot application...")
        
        # Initialize VoiceBot
        voicebot = VoiceBotUI()
        
        # Create Gradio interface
        demo = create_gradio_interface(voicebot)
        # Enable request queue for chat inference and audio processing
        demo.queue()
        
        # Launch with optimized settings
        logger.info("🌐 Launching Gradio interface...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            debug=False,
            show_error=True,
            quiet=False,
            show_api=False,
            max_threads=10
        )
        
        logger.info("✅ Gradio interface launched successfully!")

    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        print(f"\n🔴 CRITICAL ERROR: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if all required files exist (FAISS index, embeddings)")
        print("2. Verify AWS credentials are properly configured")
        print("3. Ensure all Python dependencies are installed")
        print("4. Check if ports 7860 is available")
        raise

if __name__ == "__main__":
    main()