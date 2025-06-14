import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import numpy as np
import pandas as pd
import os

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Oriserve Hindi2Hinglish Whisper model and processor
model_id = "Oriserve/Whisper-Hindi2Hinglish-Swift"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create the ASR pipeline
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if torch.cuda.is_available() else -1,
    generate_kwargs={"task": "transcribe"}
)

def transcribe_audio(audio_path):
    """
    Transcribes Hindi, Hinglish, or English audio to text.
    Args:
        audio_path (str): Path to the .wav audio file.
    Returns:
        str: Transcribed text.
    """
    # Load audio with soundfile (bypasses ffmpeg)
    audio_data, sample_rate = sf.read(audio_path)
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    # Convert to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    # Pass as dict to pipeline
    result = asr_pipe({"array": audio_data, "sampling_rate": sample_rate})
    return result["text"]

# Example usage:
# Make sure 'user_input.wav' is your recorded file from Step 1
transcribed_text = transcribe_audio('Data/user_input_1.wav')
print("Transcription:", transcribed_text)

# Save to CSV under 'Questions' column
csv_path = 'Data/transcriptions.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=['Questions'])
new_row = pd.DataFrame({'Questions': [transcribed_text]})
df = pd.concat([df, new_row], ignore_index=True)
df.to_csv(csv_path, index=False)
print(f"Saved transcription to {csv_path}")
