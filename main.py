import os
import gradio as gr
import tempfile
from modules.asr_module import ASRModule
from modules.response_gen import ResponseGenerator
import boto3
from modules.utils import Config, get_aws_credentials

class VoiceBotUI:
    def __init__(self):
        self.config = Config()
        self.asr_module = ASRModule()
        self.response_gen = ResponseGenerator()
        
        # Initialize Polly client for text-to-speech
        credentials = get_aws_credentials()
        self.polly_client = boto3.client('polly', **credentials)
        self.voice_id = self.config.get('polly', {}).get('voice_id', 'Joanna')
        self.engine = self.config.get('polly', {}).get('engine', 'neural')

    def process_audio(self, audio_path):
        """Process audio input and return response"""
        if audio_path is None:
            return "No audio input received", None
        
        # Get transcription
        transcript = self.asr_module.transcribe_audio(audio_path)
        if not transcript:
            return "Failed to transcribe audio", None
        
        # Get response
        response, source, confidence = self.response_gen.get_response(transcript)
        
        # Generate audio response
        audio_response = self.text_to_speech(response)
        
        return f"""Transcript: {transcript}
        
Response: {response}

(Source: {source}, Confidence: {confidence:.2f}%)""", audio_response

    def text_to_speech(self, text):
        """Convert text to speech using AWS Polly"""
        try:
            response = self.polly_client.synthesize_speech(
                Engine=self.engine,
                OutputFormat='mp3',
                Text=text,
                VoiceId=self.voice_id
            )
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                f.write(response['AudioStream'].read())
                return f.name
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return None

def main():
    voicebot = VoiceBotUI()
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=voicebot.process_audio,
        inputs=[
            gr.Audio(source="microphone", type="filepath", label="Speak or Upload Audio"),
        ],
        outputs=[
            gr.Textbox(label="Response"),
            gr.Audio(label="Voice Response")
        ],
        title="Voice Assistant Demo",
        description="Speak or upload an audio file to get a response. The system will transcribe your speech, find the best matching response, and read it back to you.",
        examples=[],  # Add example audio files if available
        theme="default"
    )
    
    # Launch the interface
    iface.launch(share=True)

if __name__ == "__main__":
    main()
