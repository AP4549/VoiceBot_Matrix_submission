import logging
logging.basicConfig(level=logging.INFO)
from main import main

if __name__ == "__main__":
    try:
        print("Starting VoiceBot...")
        main()
    except Exception as e:
        print(f"Error running VoiceBot: {str(e)}")
        import traceback
        traceback.print_exc()
