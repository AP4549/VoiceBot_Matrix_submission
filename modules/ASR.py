import torch
import os
import soundfile as sf
import numpy as np
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datetime import datetime
from modules.response_gen_rag import ResponseGeneratorRAG

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    # Convert relative path to absolute path if needed
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(PROJECT_ROOT, audio_path)
    
    # Read audio file using soundfile
    audio_data, sample_rate = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Convert to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Process with the ASR pipeline
    result = asr_pipe({"array": audio_data, "sampling_rate": sample_rate})
    return result["text"]

def save_transcription_to_csv(transcription, csv_path="Data/transcriptions.csv"):
    """
    Save the transcription to a CSV file with a Question column.
    Args:
        transcription (str): The transcribed text
        csv_path (str): Path to the CSV file
    Returns:
        str: Path to the saved file
    """
    # Convert to absolute path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, csv_path)
    
    print(f"Attempting to save to: {csv_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    print(f"Directory created/verified: {os.path.dirname(csv_path)}")
    
    # Create or load the CSV file
    if os.path.exists(csv_path):
        print(f"Loading existing CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"Creating new CSV file: {csv_path}")
        df = pd.DataFrame(columns=['Question'])
    
    # Add new transcription
    new_row = pd.DataFrame({'Question': [transcription]})
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved to: {csv_path}")
    
    # Verify file exists
    if os.path.exists(csv_path):
        print(f"File exists after saving: {csv_path}")
        print(f"File size: {os.path.getsize(csv_path)} bytes")
    else:
        print(f"Warning: File not found after saving: {csv_path}")
    
    return csv_path

def process_transcriptions_and_generate_responses(input_csv="Data/transcriptions.csv", output_csv="Data/qa_responses.csv"):
    """
    Process transcriptions and generate responses for each question.
    Args:
        input_csv (str): Path to the input CSV with transcriptions
        output_csv (str): Path to save the output CSV with questions and responses
    Returns:
        str: Path to the output CSV file
    """
    # Convert paths to absolute
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(PROJECT_ROOT, input_csv)
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(PROJECT_ROOT, output_csv)
    
    print(f"Processing transcriptions from: {input_csv}")
    
    # Initialize response generator
    faiss_index_path = os.path.join(PROJECT_ROOT, "Data/qa_faiss_index.bin")
    embeddings_file_path = os.path.join(PROJECT_ROOT, "Data/qa_embeddings.npy")
    response_gen = ResponseGeneratorRAG(faiss_index_path=faiss_index_path, embeddings_file_path=embeddings_file_path)
    
    # Read transcriptions
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        return None
    
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} questions to process")
    
    # Generate responses
    responses = []
    for idx, row in df.iterrows():
        question = row['Question']
        print(f"Processing question {idx + 1}/{len(df)}: {question[:50]}...")
        response, source, confidence = response_gen.get_response(question)
        responses.append({
            'Question': question,
            'Response': response,
            'Source': source,
            'Confidence': confidence
        })
    
    # Create output DataFrame
    output_df = pd.DataFrame(responses)
    
    # Save to CSV
    output_df.to_csv(output_csv, index=False)
    print(f"Responses saved to: {output_csv}")
    
    return output_csv

# Example usage:
if __name__ == "__main__":
    # First, transcribe the audio if needed
    transcribed_text = transcribe_audio('Data/user_input_1.wav')
    transcriptions_file = save_transcription_to_csv(transcribed_text)
    
    # Then process all transcriptions and generate responses
    responses_file = process_transcriptions_and_generate_responses()
    print(f"Processing complete. Responses saved to: {responses_file}")
