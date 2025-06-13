import pandas as pd
from modules.response_gen import ResponseGenerator

# Test questions in both English and Hindi
test_questions = [
    "What is a savings account?",
    "How do I open a bank account?",
    "बैंक खाता कैसे खोलें?",
    "बचत खाता क्या होता है?",
    "ऋण के लिए क्या दस्तावेज़ चाहिए?",
    "Tell me about NABARD",
    "नाबार्ड के बारे में बताएं"
]

def test_qa_system():
    print("Testing QA System...")
    response_gen = ResponseGenerator()
    
    for question in test_questions:
        print("\nQuestion:", question)
        response, source, confidence = response_gen.get_response(question)
        print(f"Source: {source}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Response: {response[:100]}...")  # Show first 100 chars

if __name__ == "__main__":
    test_qa_system()
