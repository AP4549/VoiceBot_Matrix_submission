import boto3
from typing import Optional
from botocore.exceptions import ClientError
from .utils import get_aws_credentials

class AWSTTSModule:
    def __init__(self):
        credentials = get_aws_credentials()
        self.polly_client = boto3.client('polly', **credentials)
        self.voice_configs = {
            'en': {
                'VoiceId': 'Kajal',  # Indian English voice
                'Engine': 'neural',
                'LanguageCode': 'en-IN'
            },
            'hi': {
                'VoiceId': 'Kajal',  # Using Kajal for Hindi too as it handles both well
                'Engine': 'neural',
                'LanguageCode': 'hi-IN'
            },
            'hi-Latn': {  # For Hinglish (Hindi in Latin script)
                'VoiceId': 'Kajal',  # Using Kajal for better Hinglish pronunciation
                'Engine': 'neural',
                'LanguageCode': 'hi-IN'
            }
        }

    def detect_script(self, text: str) -> str:
        """Detect if text is in Devanagari or Latin script"""
        devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        return 'Deva' if devanagari_count > 0 else 'Latn'

    def synthesize_speech(self, text: str, language: str = 'en') -> Optional[str]:
        """
        Synthesize speech using AWS Polly with enhanced language-specific configurations
        Returns: Path to the generated audio file
        """
        try:
            # Sanitize and prepare text
            text = text.strip().replace('"', "'")
            output_path = f"temp_speech_{hash(text)}.mp3"

            # Handle Hinglish and Hindi
            if language == 'hi':
                script = self.detect_script(text)
                if script == 'Latn':
                    voice_config = self.voice_configs['hi-Latn']
                else:
                    voice_config = self.voice_configs['hi']
            else:
                voice_config = self.voice_configs['en']

            # Configure Polly request
            polly_request = {
                'Engine': voice_config['Engine'],
                'LanguageCode': voice_config['LanguageCode'],
                'OutputFormat': 'mp3',
                'Text': text,
                'TextType': 'text',  # Always use plain text to avoid SSML issues
                'VoiceId': voice_config['VoiceId']
            }

            # Generate speech
            response = self.polly_client.synthesize_speech(**polly_request)
            
            # Save audio
            if "AudioStream" in response:
                with open(output_path, 'wb') as file:
                    file.write(response['AudioStream'].read())
                return output_path
                
        except ClientError as e:
            print(f"AWS Polly error: {str(e)}")
        except Exception as e:
            print(f"Error in synthesize_speech: {str(e)}")
        
        return None
