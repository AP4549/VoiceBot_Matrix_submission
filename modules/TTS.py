import pandas as pd
import asyncio
import edge_tts
import os
import platform
import subprocess
import re

def detect_hindi(text):
    # Returns True if the text contains Devanagari (Hindi) script
    return bool(re.search('[\u0900-\u097F]', text))

def play_audio(file_path='response.mp3'):
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.call(['afplay', file_path])
        elif system == 'Windows':
            os.startfile(file_path)
        else:  # Linux
            try:
                subprocess.call(['aplay', file_path])
            except FileNotFoundError:
                subprocess.call(['ffplay', '-nodisp', '-autoexit', file_path])
    except Exception as e:
        print(f"Could not play audio: {e}")

async def text_to_speech(text, output_file='response.mp3', voice='en-US-JennyNeural'):
    if not text.strip():
        print("Warning: Empty text, skipping TTS.")
        return False
    if len(text) > 400:
        print("Text too long, truncating for TTS.")
        text = text[:400]
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        print(f"Audio saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return False

if __name__ == "__main__":
    # Load CSV file with responses
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_FILENAME = os.path.join(project_root, "Data/transcriptions.csv")  # Change as needed
    RESPONSE_COLUMN = 'Response'  # Change if your column name differs

    try:
        df = pd.read_csv(CSV_FILENAME, encoding='utf-8')
    except FileNotFoundError:
        print(f"CSV file '{CSV_FILENAME}' not found. Please check the path and filename.")
        exit(1)

    responses = df[RESPONSE_COLUMN].dropna().astype(str).tolist()

    for idx, text in enumerate(responses):
        print(f"\nResponse {idx+1}: {text}\n")
        voice = 'hi-IN-NeerjaNeural' if detect_hindi(text) else 'en-US-JennyNeural'
        audio_file = f"response_{idx+1}.mp3"
        # Robust event loop handling for Windows
        try:
            asyncio.run(text_to_speech(text, audio_file, voice=voice))
        except RuntimeError as e:
            if "event loop is closed" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(text_to_speech(text, audio_file, voice=voice))
            else:
                raise
        if os.path.exists(audio_file):
            play_audio(audio_file)
        else:
            print(f"Failed to convert response {idx+1} to speech")
        input("Press Enter for the next response...")
