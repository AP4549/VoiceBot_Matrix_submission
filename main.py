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

    def process_audio(self, audio_path):
        """Process audio input and return (transcript, response, audio_response, history)"""
        if audio_path is None:
            logger.warning("No audio input received")
            return "No audio input received", "Please provide audio input to continue.", None, []
        
        try:
            logger.info(f"Processing audio: {audio_path}")
            
            # Create unique filename
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            audio_filename = f"user_input_{timestamp_str}.wav"
            local_audio_path = self.audio_dir / audio_filename
            
            # Handle different audio input types
            try:
                if isinstance(audio_path, tuple):
                    sample_rate, audio_data = audio_path
                    self._save_audio(local_audio_path, audio_data, sample_rate)
                else:
                    if not os.path.exists(audio_path):
                        logger.error(f"Audio file not found: {audio_path}")
                        return f"Error: Audio file not found at {audio_path}", "Audio file error.", None, []
                    
                    audio_data, sample_rate = sf.read(audio_path)
                    self._save_audio(local_audio_path, audio_data, sample_rate)
                
                audio_path = str(local_audio_path)
            except Exception as audio_error:
                logger.error(f"Audio processing error: {audio_error}")
                return f"Audio processing error: {str(audio_error)}", "Failed to process audio.", None, []

            # Speech recognition
            logger.info("Starting transcription...")
            transcript = self.asr_module.transcribe_audio(audio_path)
            
            if not transcript:
                return "Transcription failed", "Failed to transcribe audio", None, []
            
            clean_transcript = clean_text(transcript)
            detected_lang = self.detect_language(clean_transcript)
            format_type = 'devanagari' if '\u0900' <= clean_transcript[0] <= '\u097F' else 'latin'
            
            logger.info(f"Transcription successful: {clean_transcript}")
            
            # Generate response based on authentication state
            if self.current_user and self.access_token:
                # Use contextual RAG with Supabase
                try:
                    # Get recent conversations for context
                    recent = self.supabase.get_recent_conversations(
                        self.current_user.id,
                        self.access_token,
                        limit=5
                    )
                    
                    # Build context from recent conversations
                    context = ""
                    if recent:
                        for conv in recent:
                            context += f"User: {conv['message']}\nAssistant: {conv['response']}\n"
                        context += f"User: {clean_transcript}\n"
                    
                    # Generate response with context using RAGB
                    logger.info("Generating response with context...")
                    response, source, confidence = self.response_gen_with_context.get_response(
                        clean_transcript,
                        context=context if context else None
                    )
                    
                except Exception as context_error:
                    logger.error(f"Failed to use contextual response: {context_error}")
                    # Fallback to basic RAG
                    response, source, confidence = self.response_gen.get_response(clean_transcript)
            else:
                # Use basic RAG without context
                logger.info("Generating response without context...")
                response, source, confidence = self.response_gen.get_response(clean_transcript)
            
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                source = "fallback"
                confidence = 0.0
            
            logger.info(f"Response generated: {response[:50]}... (confidence: {confidence:.1f}%)")
            
            # Generate audio response
            logger.info("Generating audio response...")
            audio_response = self.text_to_speech(response, detected_lang)
            
            # Store in Supabase if authenticated
            try:
                if self.current_user and self.access_token:
                    conversation = self.supabase.store_conversation(
                        user_id=self.current_user.id,
                        message=clean_transcript,
                        response=response,
                        audio_url=audio_path,
                        response_audio_url=audio_response,
                        language=detected_lang,
                        confidence_score=confidence,
                        format=format_type,
                        source=source,
                        access_token=self.access_token
                    )
                    logger.info("✅ Conversation stored in Supabase")
                    
                    # Get updated conversation history
                    recent = self.supabase.get_recent_conversations(
                        self.current_user.id,
                        self.access_token,
                        limit=5
                    )
                    recent_interactions = [[conv["message"], conv["response"]] for conv in recent]
                else:
                    # For non-authenticated users, just show current interaction
                    recent_interactions = [[clean_transcript, response]]
                    
            except Exception as db_error:
                logger.error(f"Failed to store conversation: {db_error}")
                recent_interactions = [[clean_transcript, response]]
            
            logger.info("✅ Audio processing completed successfully")
            return (
                clean_transcript,    # Transcript
                response,           # Text response
                audio_response,     # Audio response
                recent_interactions # Conversation history
            )
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            error_msg = f"Processing error: {str(e)}"
            return error_msg, "An error occurred while processing your request.", None, []

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

def create_gradio_interface(voicebot):
    """Create and configure Gradio interface"""
    try:
        logger.info("Setting up Gradio interface...")
        
        def safe_process_audio(audio_path):
            """Wrapper for safe audio processing"""
            try:
                if audio_path is None:
                    return "No audio input received.", "Please provide audio input to continue.", None, []
                
                result = voicebot.process_audio(audio_path)
                if not result or len(result) != 4:
                    return "Processing failed.", "An error occurred during processing.", None, []
                
                transcript, response, audio_out, history = result
                
                # Ensure we have valid outputs
                if not transcript:
                    transcript = "Transcription failed"
                if not response:
                    response = "Response generation failed"
                if not isinstance(history, list):
                    history = []
                    
                return transcript, response, audio_out, history
                
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                return str(e), "An error occurred", None, []
                
        def handle_login(email, password):
            """Handle user login"""
            success, message = voicebot.sign_in_user(email, password)
            if success:
                return message, True, gr.update(visible=False), gr.update(visible=True)
            return message, False, gr.update(visible=True), gr.update(visible=False)
            
        def handle_logout():
            """Handle user logout"""
            success, message = voicebot.sign_out_user()
            if success:
                return message, gr.update(visible=True), gr.update(False)
            return message, gr.update(visible=True), gr.update(False)

        # Create Gradio interface
        demo = gr.Blocks()
        
        with demo:
            gr.Markdown("# 🎙️ Voice Bot Interface")
            login_msg = gr.Textbox(label="Login Status", interactive=False)
            
            # Login components
            with gr.Row(visible=True) as login_row:
                email = gr.Textbox(
                    label="Email",
                    placeholder="Enter your email"
                )
                password = gr.Textbox(
                    label="Password",
                    placeholder="Enter your password",
                    type="password"
                )
                login_btn = gr.Button("Login")
            
            # Main interface (hidden until login)
            with gr.Column(visible=False) as main_interface:
                logout_btn = gr.Button("Logout")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="🎤 Record or Upload Audio",
                            type="filepath",
                            sources=["microphone", "upload"]
                        )
                        submit_btn = gr.Button(
                            "Submit",
                            variant="primary"
                        )
                        transcript_box = gr.Textbox(
                            label="🔤 Transcript",
                            placeholder="Transcription will appear here...",
                            show_copy_button=True
                        )

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="🤖 Bot Response",
                            placeholder="Response will appear here...",
                            show_copy_button=True
                        )
                        
                        audio_output = gr.Audio(
                            label="🔊 Voice Response",
                            type="filepath",
                            autoplay=False,
                            show_download_button=True
                        )

                with gr.Row():                    
                    conversation_history = gr.Dataframe(
                        label="💬 Recent Conversations",
                        headers=["Question", "Response"],
                        wrap=True,
                        interactive=False
                    )                # Only process on submit button click
                submit_btn.click(
                    fn=safe_process_audio,
                    inputs=[audio_input],
                    outputs=[transcript_box, output_text, audio_output, conversation_history],
                    show_progress=True
                )
            
            # Connect login handlers
            login_btn.click(
                fn=handle_login,
                inputs=[email, password],
                outputs=[login_msg, gr.State(value=True), login_row, main_interface]
            )
            
            logout_btn.click(
                fn=handle_logout,
                inputs=[],
                outputs=[login_msg, login_row, main_interface]
            )

        logger.info("✅ Gradio interface setup complete")
        return demo
        
    except Exception as e:
        logger.error(f"❌ Gradio interface setup failed: {e}")
        raise

def main():
    """Main function with comprehensive error handling"""
    try:
        logger.info("🚀 Starting VoiceBot application...")
        
        # Initialize VoiceBot
        voicebot = VoiceBotUI()
        
        # Create Gradio interface
        demo = create_gradio_interface(voicebot)
        
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