import os
import unittest
import traceback
from modules.aws_asr import AWSTranscribeModule
from pathlib import Path

class TestAWSTranscribe(unittest.TestCase):
    def setUp(self):
        self.asr_module = AWSTranscribeModule()
        # Create test audio directory if it doesn't exist
        self.test_dir = Path("test_audio")
        self.test_dir.mkdir(exist_ok=True)
        # Get the first wav file from the audio directory
        audio_dir = Path("Data/userinputvoice/audio")
        wav_files = list(audio_dir.glob("*.wav"))
        if not wav_files:
            self.skipTest("No WAV files found in Data/userinputvoice/audio")
        self.sample_audio = str(wav_files[0])

    def test_audio_conversion(self):
        """Test audio format conversion"""
        print(f"\nTesting audio conversion with file: {self.sample_audio}")
        converted_path = self.asr_module.convert_audio_format(self.sample_audio)
        
        self.assertIsNotNone(converted_path, "Audio conversion failed")
        self.assertTrue(os.path.exists(converted_path), "Converted file does not exist")
        self.assertTrue(converted_path.endswith('.mp3'), "Converted file is not MP3")
        print(f"Successfully converted to: {converted_path}")
        
        # Cleanup
        if os.path.exists(converted_path):
            os.remove(converted_path)

    def test_transcribe_audio(self):
        """Test audio transcription"""
        try:
            print(f"\nTesting transcription with file: {self.sample_audio}")
            transcript = self.asr_module.transcribe_audio(self.sample_audio)
            
            self.assertIsNotNone(transcript, "Transcription failed")
            self.assertIsInstance(transcript, str, "Transcript should be a string")
            self.assertTrue(len(transcript) > 0, "Transcript is empty")
            
            print(f"Transcript: {transcript}")
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            raise

    def tearDown(self):
        # Cleanup test directory
        if self.test_dir.exists():
            for file in self.test_dir.glob("*"):
                file.unlink()
            self.test_dir.rmdir()

if __name__ == '__main__':
    unittest.main()
