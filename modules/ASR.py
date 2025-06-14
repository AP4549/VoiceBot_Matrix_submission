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
if __name__ == "__main__":
    audio_file_path = 'Data/user_input_1.wav' # Path to the audio file for standalone testing
    
    # Ensure the Data directory exists for where the audio file should be
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    full_audio_path = os.path.join(project_root, audio_file_path)

    if os.path.exists(full_audio_path):
        transcribed_text = transcribe_audio(full_audio_path)
        print("Transcription:", transcribed_text)

        # If you still want to save to CSV when running ASR.py directly, uncomment these:
        # transcriptions_file = save_transcription_to_csv(transcribed_text, os.path.join(project_root, "Data/transcriptions.csv"))
        # print(f"Transcription saved to: {transcriptions_file}")

        # If you still want to process responses when running ASR.py directly, uncomment these:
        # responses_file = process_transcriptions_and_generate_responses(input_csv=transcriptions_file, output_csv=os.path.join(project_root, "Data/qa_responses.csv"))
        # print(f"Processing complete. Responses saved to: {responses_file}")
    else:
        print(f"Error: Audio file not found at '{full_audio_path}'.")
        print("Please ensure you have recorded audio or placed 'user_input_1.wav' in the 'Data' directory relative to your project root.")
        print("When running the full VoiceBot MATRIX, the audio is recorded automatically.")
