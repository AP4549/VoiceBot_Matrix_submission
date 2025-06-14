# Matrix VoiceBot Submission

A voice-enabled chatbot that matches user questions against a preloaded QA dataset and optionally uses AWS Bedrock for fallback responses.

## Features

- **Round 1: CSV-Based QA Matching**
  - Fuzzy matching against QA dataset
  - Optional AWS Bedrock fallback
  - Batch processing via `test.csv`

- **Round 2: Live Voice Demo**
  - Real-time audio input via microphone
  - File upload support
  - AWS Transcribe for speech-to-text
  - AWS Polly for text-to-speech responses
  - Interactive Gradio web interface

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure AWS credentials:
   - Create a `.env` file:
     ```
     AWS_ACCESS_KEY_ID=your_access_key
     AWS_SECRET_ACCESS_KEY=your_secret_key
     ```

## Using the Gradio Interface

The system now includes a Gradio web interface for easy interaction with the QA system.

1. Start the interface:
   ```bash
   python run_inference.py
   ```

2. The interface provides two modes:
   - **Batch Process**: Upload a CSV file with questions (must have a 'Questions' column)
   - **Interactive QA**: Type individual questions and get immediate answers

3. Sample CSV format (see `Data/test.csv` for an example):
   ```
   Questions
   What are the typical documents required to open a new savings account?
   How do I apply for a credit card?
   ```

4. The system will provide responses along with information about the source (dataset or LLM) and confidence level.
     AWS_SESSION_TOKEN=your_session_token  # Optional
     ```

3. Update configuration:
   - Edit `config/config.yaml` with your settings:
     - AWS region
     - S3 bucket name
     - Model preferences
     - Matching thresholds

## Data Structure

1. QA Dataset (`data/qa_dataset.csv`):
   ```csv
   Question,Response
   "What are your hours?","We are open 9 AM to 5 PM Monday through Friday."
   ```

2. Test Input (`data/test.csv`):
   ```csv
   Questions
   "When do you open?"
   ```

## Usage

### Round 1: CSV Processing

```bash
python run_inference.py
```

This will:
1. Read questions from `test.csv`
2. Match against `qa_dataset.csv`
3. Generate responses using AWS Bedrock if needed
4. Save results to `output/output.csv`

### Round 2: Voice Interface

```bash
python main.py
```

This will:
1. Launch Gradio web interface
2. Enable microphone/file input
3. Process audio through AWS services
4. Display text response and play audio

## AWS Setup

1. Required Services:
   - AWS Transcribe
   - AWS Polly
   - AWS Bedrock (optional)
   - S3 bucket for audio files

2. IAM Permissions:
   - Transcribe access
   - Polly access
   - S3 read/write
   - Bedrock model access

## Error Handling

- Audio transcription failures
- Network connectivity issues
- Low confidence matches
- AWS service quotas

## Performance Notes

- Fuzzy matching threshold: 80%
- Audio sampling rate: 16kHz
- Response generation time: ~2-3s
- Voice synthesis quality: Neural engine

## Development

- Python 3.8+
- Code formatting: Black
- Type hints included
- Modular architecture
- Easy to extend

## License

MIT License
