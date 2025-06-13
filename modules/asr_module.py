import os
import time
import boto3
import tempfile
from typing import Dict, Optional
from .utils import Config, get_aws_credentials

class ASRModule:
    def __init__(self):
        self.config = Config()
        credentials = get_aws_credentials()
        self.transcribe_client = boto3.client('transcribe', **credentials)
        self.s3_client = boto3.client('s3', **credentials)
        self.bucket_name = self.config.get('s3_bucket')

    def upload_to_s3(self, audio_file: str) -> str:
        """Upload audio file to S3 and return the S3 URI."""
        file_name = os.path.basename(audio_file)
        s3_key = f"uploads/{file_name}"
        
        try:
            self.s3_client.upload_file(audio_file, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            raise

    def start_transcription_job(self, s3_uri: str, job_name: str) -> None:
        """Start an AWS Transcribe job."""
        try:
            self.transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': s3_uri},
                MediaFormat='mp3',  # Adjust based on input format
                LanguageCode=self.config.get('transcribe', {}).get('language_code', 'en-US'),
                Settings={
                    'ShowSpeakerLabels': False,
                    'MaxSpeakerLabels': 1
                }
            )
        except Exception as e:
            print(f"Error starting transcription job: {e}")
            raise

    def get_transcription_result(self, job_name: str) -> Optional[str]:
        """Get the result of a transcription job."""
        try:
            while True:
                result = self.transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                status = result['TranscriptionJob']['TranscriptionJobStatus']
                
                if status == 'COMPLETED':
                    transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    response = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=transcript_uri.split(self.bucket_name + '/')[-1]
                    )
                    transcript = response['Body'].read().decode('utf-8')
                    return transcript
                elif status == 'FAILED':
                    print(f"Transcription job failed: {result['TranscriptionJob'].get('FailureReason', 'Unknown error')}")
                    return None
                
                time.sleep(5)  # Wait before checking again
                
        except Exception as e:
            print(f"Error getting transcription result: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Main method to transcribe an audio file."""
        try:
            # Generate a unique job name
            job_name = f"transcribe_{os.path.basename(audio_path)}_{int(time.time())}"
            
            # Upload to S3
            s3_uri = self.upload_to_s3(audio_path)
            
            # Start transcription
            self.start_transcription_job(s3_uri, job_name)
            
            # Get results
            transcript = self.get_transcription_result(job_name)
            
            return transcript
            
        except Exception as e:
            print(f"Error in transcription process: {e}")
            return None
