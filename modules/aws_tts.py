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
                'VoiceId': 'Aditi',  # Hindi voice
                'Engine': 'standard',  # Aditi only supports standard engine
                'LanguageCode': 'hi-IN'
            },
            'hi-Latn': {  # For Hinglish (Hindi in Latin script)
                'VoiceId': 'Kajal',  # Using Kajal for better Hinglish pronunciation
                'Engine': 'neural',
                'LanguageCode': 'en-IN'
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
            # Handle Hinglish (Hindi in Latin script)
            if language == 'hi':
                script = self.detect_script(text)
                if script == 'Latn':
                    # Use hi-Latn config for Hinglish
                    voice_config = self.voice_configs['hi-Latn']
                else:
                    # Use standard Hindi config for Devanagari
                    voice_config = self.voice_configs['hi']
            else:
                # Default to English configuration
                voice_config = self.voice_configs['en']
            
            # Add SSML for better pronunciation if needed
            if language == 'hi' and self.detect_script(text) == 'Latn':
                # Wrap Hinglish text in SSML to improve pronunciation
                text = f'<speak><amazon:domain name="conversational">{text}</amazon:domain></speak>'
                
            response = self.polly_client.synthesize_speech(
                Text=text,
                TextType='ssml' if '<speak>' in text else 'text',
                OutputFormat='mp3',
                VoiceId=voice_config['VoiceId'],
                Engine=voice_config['Engine'],
                LanguageCode=voice_config['LanguageCode']
            )
            
            if 'AudioStream' in response:
                # Create a temporary file to store the audio
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    f.write(response['AudioStream'].read())
                    return f.name
                    
            return None
            
        except ClientError as e:
            print(f"Error in text-to-speech: {str(e)}")
            return None
