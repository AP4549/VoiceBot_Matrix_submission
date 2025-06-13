import pandas as pd
from modules.response_gen import ResponseGenerator

# Test questions in both English and Hindi
test_questions = [
    "Given my current financial situation with a modest income and a desire to save for a down payment on a house within the next five years, what specific savings strategies would you recommend, and what potential risks should I be aware of?",
    "मैं अपनी छोटी सी दुकान के लिए बैंक से व्यवसाय ऋण लेना चाहता हूँ। मुझे इस प्रक्रिया में किन चुनौतियों का सामना करना पड़ सकता है, और मैं अपनी अनुमोदन संभावनाओं को कैसे बेहतर बना सकता हूँ?"
]

def test_qa_system():
    print("Testing QA System...")
    response_gen = ResponseGenerator()
    
    for question in test_questions:
        print("\nQuestion:", question)
        response, source, confidence = response_gen.get_response(question)
        print(f"Source: {source}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Response: {response[:500]}...")  # Show first 100 chars

if __name__ == "__main__":
    test_qa_system()
