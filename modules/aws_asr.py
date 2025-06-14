import os
import time
import boto3
import json
import tempfile
import pandas as pd
from pydub import AudioSegment
from typing import Optional
from .utils import Config, get_aws_credentials
import requests

class AWSTranscribeModule:
    def __init__(self):
        self.config = Config()
        credentials = get_aws_credentials()
        self.transcribe_client = boto3.client('transcribe', **credentials)
        self.s3_client = boto3.client('s3', **credentials)
        self.bucket_name = self.config.get('s3_bucket', 'voicebot-matrix-hackathon')

    def convert_audio_format(self, audio_path: str) -> Optional[str]:
        """Convert audio to format compatible with AWS Transcribe."""
        temp_file = None
        try:
            # Load audio using pydub - ensure proper cleanup
            with open(audio_path, 'rb') as audio_src:
                audio = AudioSegment.from_file(audio_src)
                
                # Convert to mp3 with specific parameters
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                audio.export(
                    temp_file.name, 
                    format='mp3', 
                    parameters=["-ar", "16000"]
                )
                temp_file.close()  # Close file explicitly
                return temp_file.name
                
        except Exception as e:
            print(f"Error converting audio format: {e}")
            if temp_file and hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return None

    def cleanup_s3_object(self, s3_key: str) -> None:
        """Clean up temporary audio file from S3 bucket."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            print(f"Cleaned up S3 object: {s3_key}")
        except Exception as e:
            print(f"Warning: Failed to cleanup S3 object {s3_key}: {e}")
            # Non-critical error, don't raise

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using AWS Transcribe."""
        try:
            # Convert audio format if needed
            processed_audio_path = self.convert_audio_format(audio_path)
            if not processed_audio_path:
                return None

            # Generate unique job name
            timestamp = str(int(time.time()))
            job_name = f"transcription_{timestamp}"
            s3_key = f"audio/{os.path.basename(processed_audio_path)}"

            # Upload to S3
            with open(processed_audio_path, 'rb') as audio_file:
                self.s3_client.upload_fileobj(audio_file, self.bucket_name, s3_key)            # Start transcription job with language identification
            print(f"Starting transcription job: {job_name}")
            try:
                response = self.transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={'MediaFileUri': f"s3://{self.bucket_name}/{s3_key}"},
                    MediaFormat='mp3',
                    IdentifyLanguage=True,
                    LanguageOptions=['en-US', 'hi-IN']
                )
                print(f"Transcription job created successfully: {job_name}")
            except Exception as e:
                print(f"Error creating transcription job: {e}")
                raise

            # Wait for completion with timeout
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                try:
                    status = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                    current_status = status['TranscriptionJob']['TranscriptionJobStatus']
                    print(f"Job status: {current_status}")
                    
                    if current_status == 'COMPLETED':
                        break
                    elif current_status == 'FAILED':
                        failure_reason = status['TranscriptionJob'].get('FailureReason', 'Unknown error')
                        print(f"Transcription job failed: {failure_reason}")
                        return None
                    
                    time.sleep(5)
                except Exception as e:
                    print(f"Error checking job status: {e}")
                    time.sleep(5)
                    continue

            if 'status' in locals() and status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                # Get transcription result
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                response = requests.get(transcript_uri)
                data = response.json()
                
                # Clean up
                self.cleanup_s3_object(s3_key)
                os.unlink(processed_audio_path)

                # Return the transcribed text
                return data['results']['transcripts'][0]['transcript']
            else:
                print(f"Transcription failed: {status['TranscriptionJob'].get('FailureReason', 'Unknown error')}")
                return None

        except Exception as e:
            print(f"Error in transcription process: {e}")
            return None
            
    def save_transcription(self, transcription: str, filepath: str = "Data/userinputvoice/user_inputs.csv"):
        """Save transcription to CSV file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create or load existing CSV
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
            else:
                df = pd.DataFrame(columns=['Timestamp', 'Question'])
            
            # Add new transcription
            new_row = pd.DataFrame([{
                'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'Question': transcription
            }])
            
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(filepath, index=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving transcription: {e}")
            return False
