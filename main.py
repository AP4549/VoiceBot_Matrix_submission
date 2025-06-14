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

class VoiceBotUI:
    def __init__(self):
        self.config = Config()
        self.asr_module = AWSTranscribeModule()
        self.tts_module = AWSTTSModule()
        
        # Define paths for FAISS artifacts
        faiss_index_path = "Data/qa_faiss_index.bin"
        embeddings_file_path = "Data/qa_embeddings.npy"
        
        # Initialize RAG response generator
        self.response_gen = ResponseGeneratorRAG(
            faiss_index_path=faiss_index_path, 
            embeddings_file_path=embeddings_file_path
        )
        
        # Create output directories if they don't exist
        self.input_dir = Path("Data/userinputvoice")
        self.output_dir = Path("Data/voicebotoutput")
        self.audio_dir = Path("Data/userinputvoice/audio")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self.input_csv_path = self.input_dir / "user_inputs.csv"
        self.output_csv_path = self.output_dir / "voice_responses.csv"
        
        if not self.input_csv_path.exists():
            pd.DataFrame(columns=["Timestamp", "Question", "AudioFilePath", "Language"]).to_csv(self.input_csv_path, index=False)
        
        if not self.output_csv_path.exists():
            pd.DataFrame(columns=["Timestamp", "Question", "Response", "Source", "Confidence", "Language"]).to_csv(self.output_csv_path, index=False)
        
        try:
            self.input_df = pd.read_csv(self.input_csv_path)
        except pd.errors.EmptyDataError:
            self.input_df = pd.DataFrame(columns=["Timestamp", "Question", "AudioFilePath", "Language"])
            
        try:
            self.output_df = pd.read_csv(self.output_csv_path)
        except pd.errors.EmptyDataError:
            self.output_df = pd.DataFrame(columns=["Timestamp", "Question", "Response", "Source", "Confidence", "Language"])

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            lang = detect(text)
            if lang not in ['en', 'hi']:
                return 'hi'  # Default to Hindi for unsupported languages
            return lang
        except:
            return 'hi'

    def _is_hinglish(self, text: str) -> bool:
        """Check if the text is Hinglish (Hindi written in Roman script)."""
        if not text:
            return False
        devanagari_count = sum('\u0900' <= c <= '\u097F' for c in text)
        latin_count = sum('a' <= c.lower() <= 'z' for c in text if c.isalpha())
        return devanagari_count == 0 and latin_count > 5

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages using AWS Translate."""
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
            print(f"Translation error: {e}")
            return text    
    def process_audio(self, audio_path):
        """Process audio input and return (transcript, response, audio_response, history)"""
        if audio_path is None:
            return "No audio input received", None, []
        
        try:
            # Create a unique filename for the audio
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            audio_filename = f"user_input_{timestamp_str}.wav"
            local_audio_path = self.audio_dir / audio_filename
            
            # Handle the audio file
            try:
                if isinstance(audio_path, tuple):
                    sample_rate, audio_data = audio_path
                    if isinstance(audio_data, list):
                        audio_data = np.array(audio_data)
                else:
                    if not os.path.exists(audio_path):
                        return f"Error: Audio file not found at {audio_path}", None, []
                    audio_data, sample_rate = sf.read(audio_path)
                
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                audio_data = np.float32(audio_data)
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
                os.makedirs(os.path.dirname(str(local_audio_path)), exist_ok=True)
                sf.write(str(local_audio_path), audio_data, sample_rate)
                print(f"Saved audio input to {local_audio_path}")
                
            except Exception as audio_error:
                print(f"Error handling audio: {audio_error}")
                return f"Error processing audio: {str(audio_error)}", None, []

            # Use AWS Transcribe for ASR
            transcript = self.asr_module.transcribe_audio(str(local_audio_path))
            
            if not transcript:
                return "Failed to transcribe audio", None, []
            
            # Clean transcript and detect language
            clean_transcript = clean_text(transcript)
            detected_lang = self.response_gen.detect_language(clean_transcript)
            is_hinglish = self.response_gen._is_hinglish(clean_transcript)
            
            # Save input to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_input = pd.DataFrame({
                "Timestamp": [timestamp],
                "Question": [clean_transcript],
                "AudioFilePath": [str(local_audio_path)],
                "Language": [detected_lang],
                "Format": ["Hinglish" if is_hinglish else ("Hindi" if detected_lang == 'hi' else "English")]
            })
            
            self.input_df = pd.concat([self.input_df, new_input], ignore_index=True)
            self.input_df.to_csv(self.input_csv_path, index=False, encoding='utf-8-sig')
            
            # Get response
            response, source, confidence = self.response_gen.get_response(clean_transcript)
            
            # Save output to CSV
            new_output = pd.DataFrame({
                "Timestamp": [timestamp],
                "Question": [clean_transcript],
                "Response": [response],
                "Source": [source],
                "Confidence": [confidence],
                "Language": [detected_lang],
                "Format": ["Hinglish" if is_hinglish else ("Hindi" if detected_lang == 'hi' else "English")]
            })
              # Ensure all columns exist in both DataFrames
            for col in new_output.columns:
                if col not in self.output_df.columns:
                    self.output_df[col] = pd.NA
            
            self.output_df = pd.concat([self.output_df, new_output], ignore_index=True)
            self.output_df.to_csv(self.output_csv_path, index=False, encoding='utf-8-sig')
            
            # Generate audio response
            audio_response = self.text_to_speech(response, detected_lang)
            
            # Create a dataframe with recent interactions for display
            recent_interactions = self.output_df.tail(5)[["Question", "Response", "Source", "Confidence", "Language", "Format"]].values.tolist()
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return f"Error: {str(e)}", None, []
        return (
            clean_transcript,  # For transcript box
            response,         # For response box, without metadata
            audio_response,   # For audio output
            self.output_df[["Question", "Response"]].tail(5).values.tolist()  # For conversation history
        )

    def text_to_speech(self, text: str, language: str = 'en') -> Optional[str]:
        """Convert text to speech using AWS Polly through TTS module"""
        if not text or not isinstance(text, str):
            print("Warning: Invalid text for TTS")
            return None
        
        return self.tts_module.synthesize_speech(text, language)

def main():
    try:
        print("Initializing VoiceBot...")
        voicebot = VoiceBotUI()
        print("VoiceBot initialized successfully!")

        print("Setting up Gradio interface...")
        with gr.Blocks() as demo:
            gr.Markdown("# Voice Chat Bot")
            gr.Markdown("🎙️ Record or upload a question. The bot will transcribe it, generate a response, and speak it back.")

            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Record or Upload your question",
                        sources=["microphone", "upload"],
                        type="filepath"
                    )
                    transcript_box = gr.Textbox(
                        label="Transcript",
                        placeholder="Your speech will be transcribed here...",
                        lines=2
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(
                        label="Bot Response",
                        placeholder="Response will appear here...",
                        lines=5
                    )
                    audio_output = gr.Audio(
                        label="Voice Response",
                        type="filepath",
                        autoplay=False
                    )

            conversation_history = gr.Dataframe(
                label="Recent Conversations",
                headers=["Question", "Response"],
                wrap=True
            )

            def safe_process_audio(audio_path):
                try:
                    if audio_path is None:
                        return "No audio input received.", "No audio input received.", None, []
                    
                    transcript, response, audio_out, history = voicebot.process_audio(audio_path)
                    if not transcript:
                        return "Failed to transcribe audio.", "Failed to process audio.", None, []
                        
                    return transcript, response, audio_out, history
                        
                except Exception as e:
                    print(f"Error in audio processing: {e}")
                    return str(e), "An error occurred.", None, []

            submit_btn.click(
                fn=safe_process_audio,
                inputs=[audio_input],
                outputs=[transcript_box, output_text, audio_output, conversation_history]
            )

        print("Launching Gradio interface...")
        demo.launch(share=False)
        print("Gradio interface launched successfully!")

    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()