import boto3
import json

def test_aws_services():
    print("Testing AWS Services...")
    
    # Test S3
    try:
        s3 = boto3.client('s3')
        buckets = s3.list_buckets()
        print("✅ S3 Access Successful", buckets)
    except Exception as e:
        print("❌ S3 Error:", str(e))
    
    # Test Transcribe
    try:
        transcribe = boto3.client('transcribe')
        transcribe.list_transcription_jobs(MaxResults=1)
        print("✅ Transcribe Access Successful")
    except Exception as e:
        print("❌ Transcribe Error:", str(e))
    
    # Test Polly
    try:
        polly = boto3.client('polly')
        voices = polly.describe_voices(LanguageCode='en-US')
        print("✅ Polly Access Successful")
    except Exception as e:
        print("❌ Polly Error:", str(e))
    
    # Test Bedrock
    try:
        bedrock = boto3.client('bedrock-runtime')
        print("✅ Bedrock Client Created")
    except Exception as e:
        print("❌ Bedrock Error:", str(e))

if __name__ == "__main__":
    test_aws_services()
