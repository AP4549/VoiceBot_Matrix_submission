import sounddevice as sd
import wavio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from TTS.api import TTS
import platform
import subprocess
import re
import numpy as np
import os
import random # For random speaker selection

# Import the RAG components and Config
from modules.response_gen_rag import ResponseGeneratorRAG
from modules.utils import Config, load_qa_dataset # Need load_qa_dataset for RAG init, and Config


# ----------------------------
# Audio Recording Functions
# ----------------------------
def record_audio(duration=5, fs=16000):
    """Record audio from microphone and return as numpy array"""
    print("\nRecording... Speak now ({} seconds)".format(duration))
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    # Play back user's recorded audio
    print("Playing back your recording...")
    sd.play(recording, fs)
    sd.wait() # Wait for playback to finish
    return recording, fs

# ----------------------------
# Speech Recognition (ASR)
# ----------------------------
class SpeechRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load multilingual ASR model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "Oriserve/Whisper-Hindi2Hinglish-Swift",
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained("Oriserve/Whisper-Hindi2Hinglish-Swift")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=self.torch_dtype,
        )

    def transcribe(self, audio_array, sample_rate):
        """Transcribe audio directly from numpy array"""
        # Ensure audio is mono (single channel) if it's multi-channel
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1) # Convert to mono by averaging channels

        audio_dict = {
            "array": audio_array.astype(np.float32) / 32767.0,  # Normalize to [-1, 1]
            "sampling_rate": sample_rate
        }
        return self.pipe(audio_dict)["text"]

# ----------------------------
# Response Generation (Advanced RAG)
# ----------------------------
class ResponseGenerator:
    def __init__(self):
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define paths for FAISS artifacts in the Data directory
        faiss_index_path = os.path.join(project_root, "Data", "qa_faiss_index.bin")
        embeddings_file_path = os.path.join(project_root, "Data", "qa_embeddings.npy")
        
        # Initialize RAG response generator
        self.rag = ResponseGeneratorRAG(
            faiss_index_path=faiss_index_path,
            embeddings_file_path=embeddings_file_path
        )
    
    def get_response(self, text):
        """Get response using the RAG model"""
        response, source, confidence = self.rag.get_response(text)
        print(f"Source: {source} | Confidence: {confidence:.2f}%")
        return response

# ----------------------------
# Text-to-Speech (TTS)
# ----------------------------
class VoiceSynthesizer:
    def __init__(self):
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
        
        # Randomly select a single speaker at initialization
        if not self.tts.speakers:
            print("Warning: No speakers found for the TTS model. TTS might not work.")
            self.speaker = None
        else:
            self.speaker = random.choice(self.tts.speakers)
            print(f"Using TTS speaker: {self.speaker}\n") # Added newline for better readability
    
    def speak(self, text):
        """Convert text to speech and play immediately"""
        if not self.speaker:
            print("Cannot speak: No TTS speaker selected.")
            return

        # Detect language for TTS (Coqui uses "hin" for Hindi, "en" for English)
        language = "hin" if re.search('[\u0900-\u097F]', text) else "en"
        
        # Synthesize speech directly to audio array
        try:
            audio_array, sample_rate = self.tts.tts(
                text=text,
                language=language,
                speaker=self.speaker
            )
            
            # Play audio in real-time
            print("Playing AI response...")
            sd.play(audio_array, sample_rate)
            sd.wait() # Wait for playback to finish
            
        except Exception as e:
            print(f"Error during TTS synthesis or playback: {e}")

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    # Initialize components
    recognizer = SpeechRecognizer()
    response_gen = ResponseGenerator() # Now uses RAG
    tts = VoiceSynthesizer()

    print("VoiceBot MATRIX - Ready to assist. Press Ctrl+C to exit.")

    while True:
        try:
            # Step 1: Record audio
            audio_data, sample_rate = record_audio(duration=5)
            
            # Step 2: Speech-to-text
            question = recognizer.transcribe(audio_data, sample_rate)
            if not question.strip():
                print("No clear speech detected. Please try again.")
                continue
            print(f"\nYou asked: {question}")
            
            # Step 3: Generate response (using RAG)
            response = response_gen.get_response(question)
            print(f"Bot response: {response}")
            
            # Step 4: Text-to-speech
            tts.speak(response)
            
        except KeyboardInterrupt:
            print("\nExiting VoiceBot MATRIX...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("Please ensure all dependencies are correctly installed and configured.")

if __name__ == "__main__":
    main()
