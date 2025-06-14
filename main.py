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
import boto3
from modules.utils import Config, get_aws_credentials, clean_text
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from modules.aws_tts import AWSTTSModule
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
            
            # Initialize modules with error handling
            self._initialize_modules()
            
            # Setup directories and CSV files
            self._setup_directories()
            self._initialize_csv_files()
            
            logger.info("✅ VoiceBot initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VoiceBot: {e}")
            raise

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
            
            # Initialize RAG response generator
            self.response_gen = ResponseGeneratorRAG(
                faiss_index_path=faiss_index_path, 
                embeddings_file_path=embeddings_file_path
            )
            logger.info("✅ RAG response generator initialized")
            
        except Exception as e:
            logger.error(f"❌ Module initialization failed: {e}")
            raise

    def _setup_directories(self):
        """Create necessary directories"""
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

    def _initialize_csv_files(self):
        """Initialize CSV files for data storage"""
        try:
            self.input_csv_path = self.input_dir / "user_inputs.csv"
            self.output_csv_path = self.output_dir / "voice_responses.csv"
            
            # Define column structures
            input_columns = ["Timestamp", "Question", "AudioFilePath", "Language", "Format"]
            output_columns = ["Timestamp", "Question", "Response", "Source", "Confidence", "Language", "Format"]
            
            # Initialize input CSV
            if not self.input_csv_path.exists():
                pd.DataFrame(columns=input_columns).to_csv(self.input_csv_path, index=False)
                logger.info("✅ Created new input CSV file")
            
            # Initialize output CSV
            if not self.output_csv_path.exists():
                pd.DataFrame(columns=output_columns).to_csv(self.output_csv_path, index=False)
                logger.info("✅ Created new output CSV file")
            
            # Load existing data
            try:
                self.input_df = pd.read_csv(self.input_csv_path)
                if self.input_df.empty:
                    self.input_df = pd.DataFrame(columns=input_columns)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                self.input_df = pd.DataFrame(columns=input_columns)
                
            try:
                self.output_df = pd.read_csv(self.output_csv_path)
                if self.output_df.empty:
                    self.output_df = pd.DataFrame(columns=output_columns)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                self.output_df = pd.DataFrame(columns=output_columns)
            
            logger.info("✅ CSV files initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ CSV initialization failed: {e}")
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
                    # Handle (sample_rate, audio_data) tuple
                    sample_rate, audio_data = audio_path
                    self._save_audio(local_audio_path, audio_data, sample_rate)
                else:
                    # Handle file path
                    if not os.path.exists(audio_path):
                        logger.error(f"Audio file not found: {audio_path}")
                        return f"Error: Audio file not found at {audio_path}", "Audio file error.", None, []
                    
                    # Read and save audio
                    audio_data, sample_rate = sf.read(audio_path)
                    self._save_audio(local_audio_path, audio_data, sample_rate)
                
            except Exception as audio_error:
                logger.error(f"Audio handling error: {audio_error}")
                return f"Error processing audio: {str(audio_error)}", "Audio processing failed.", None, []

            # Transcribe audio using AWS Transcribe
            logger.info("Starting transcription...")
            transcript = self.asr_module.transcribe_audio(str(local_audio_path))
            
            if not transcript or not transcript.strip():
                logger.warning("Transcription failed or empty")
                return "Failed to transcribe audio", "Could not understand the audio. Please try again.", None, []
            
            logger.info(f"Transcription successful: {transcript[:50]}...")
            
            # Clean and analyze transcript
            clean_transcript = clean_text(transcript)
            detected_lang = self.detect_language(clean_transcript)
            is_hinglish = self._is_hinglish(clean_transcript)
            
            # Determine format
            format_type = "Hinglish" if is_hinglish else ("Hindi" if detected_lang == 'hi' else "English")
            
            # Save input to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_input = pd.DataFrame({
                "Timestamp": [timestamp],
                "Question": [clean_transcript],
                "AudioFilePath": [str(local_audio_path)],
                "Language": [detected_lang],
                "Format": [format_type]
            })
            
            # Safely update input DataFrame
            try:
                self.input_df = pd.concat([self.input_df, new_input], ignore_index=True)
                self.input_df.to_csv(self.input_csv_path, index=False, encoding='utf-8-sig')
                logger.info("✅ Input data saved to CSV")
            except Exception as csv_error:
                logger.warning(f"Failed to save input CSV: {csv_error}")
            
            # Get response from RAG system
            logger.info("Generating response...")
            response, source, confidence = self.response_gen.get_response(clean_transcript)
            
            if not response:
                response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                source = "fallback"
                confidence = 0.0
            
            logger.info(f"Response generated: {response[:50]}... (confidence: {confidence:.1f}%)")
            
            # Save output to CSV
            new_output = pd.DataFrame({
                "Timestamp": [timestamp],
                "Question": [clean_transcript],
                "Response": [response],
                "Source": [source],
                "Confidence": [confidence],
                "Language": [detected_lang],
                "Format": [format_type]
            })
            
            # Safely update output DataFrame
            try:
                # Ensure all columns exist
                for col in new_output.columns:
                    if col not in self.output_df.columns:
                        self.output_df[col] = pd.NA
                
                self.output_df = pd.concat([self.output_df, new_output], ignore_index=True)
                self.output_df.to_csv(self.output_csv_path, index=False, encoding='utf-8-sig')
                logger.info("✅ Output data saved to CSV")
            except Exception as csv_error:
                logger.warning(f"Failed to save output CSV: {csv_error}")
            
            # Generate audio response
            logger.info("Generating audio response...")
            audio_response = self.text_to_speech(response, detected_lang)
            
            # Create conversation history for display
            try:
                recent_interactions = self.output_df[["Question", "Response"]].tail(5).values.tolist()
            except:
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
                logger.error(f"Safe processing error: {e}")
                return str(e), "An error occurred while processing your request.", None, []

        # Create Gradio interface
        with gr.Blocks(
            title="VoiceBot - AI Banking Assistant",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as demo:
            
            gr.Markdown(
                """
                # 🤖 VoiceBot - AI Banking Assistant
                
                🎙️ **Record or upload your banking question** - The bot will transcribe it, generate a helpful response, and speak it back to you.
                
                **Supported Languages:** English, Hindi, and Hinglish
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="🎙️ Record or Upload Your Question",
                        sources=["microphone", "upload"],
                        type="filepath",
                        show_download_button=False
                    )
                    
                    transcript_box = gr.Textbox(
                        label="📝 Transcript",
                        placeholder="Your speech will be transcribed here...",
                        lines=3,
                        max_lines=5,
                        show_copy_button=True
                    )
                    
                    submit_btn = gr.Button(
                        "🚀 Process Audio", 
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="🤖 Bot Response",
                        placeholder="Response will appear here...",
                        lines=6,
                        max_lines=10,
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
                    height=300,
                    interactive=False
                )

            # Connect the interface
            submit_btn.click(
                fn=safe_process_audio,
                inputs=[audio_input],
                outputs=[transcript_box, output_text, audio_output, conversation_history],
                show_progress=True
            )
            
            # Auto-submit when audio is uploaded
            audio_input.change(
                fn=safe_process_audio,
                inputs=[audio_input],
                outputs=[transcript_box, output_text, audio_output, conversation_history],
                show_progress=True
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