import pandas as pd
from TTS.api import TTS
import platform
import os
import subprocess
import re
import random

# Function to detect Hindi text (Devanagari script)
def detect_hindi(text):
    return bool(re.search('[\u0900-\u097F]', text))

# Function to play audio cross-platform
def play_audio(file_path):
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.call(['afplay', file_path])
        elif system == 'Windows':
            os.startfile(file_path)
        else:  # Linux
            subprocess.call(['aplay', file_path])
    except Exception as e:
        print(f"Could not play audio: {e}")

# Load responses from CSV
CSV_FILENAME = 'Data/transcriptions.csv'   # Change to your actual CSV file
RESPONSE_COLUMN = 'Response'             # Change if your column is named differently

try:
    df = pd.read_csv(CSV_FILENAME, encoding='utf-8')
except FileNotFoundError:
    print(f"CSV file '{CSV_FILENAME}' not found. Please check the path and filename.")
    exit(1)

responses = df[RESPONSE_COLUMN].dropna().astype(str).tolist()

# Initialize Coqui TTS model (multilingual, supports Hindi and English)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
print("Available speakers:", tts.speakers)
# Randomly select a single speaker for all responses
chosen_speaker = random.choice(tts.speakers)
print(f"Chosen speaker for all responses: {chosen_speaker}")

# Main loop: Synthesize and play each response
for idx, text in enumerate(responses):
    print(f"\nResponse {idx+1}: {text}\n")
    language = "hin" if detect_hindi(text) else "en"
    audio_file = f"response_{idx+1}.wav"
    try:
        tts.tts_to_file(text=text, file_path=audio_file, language=language, speaker=chosen_speaker)
        print(f"Audio saved to {audio_file}")
        play_audio(audio_file)
    except Exception as e:
        print(f"Failed to convert response {idx+1} to speech: {e}")
    input("Press Enter for the next response...")

