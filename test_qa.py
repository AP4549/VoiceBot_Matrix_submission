import pandas as pd
from modules.response_gen_rag import ResponseGeneratorRAG

# Test questions in both English and Hindi
test_questions = [
    "What are the typical documents required to open a new savings account for a student?",
    "ऑनलाइन बैंकिंग के माध्यम से मैं कौन-कौन सी सेवाएं उपयोग कर सकता हूँ?"
]

def test_qa_system():
    print("Testing QA System...")
    response_gen = ResponseGeneratorRAG()
    
    for question in test_questions:
        print("\n--- New Question ---")
        print("Question:", question)

        # Get raw FAISS matches (semantic search from dataset)
        faiss_matches = response_gen.find_best_matches_faiss(question)
        print("FAISS (Dataset) Top Matches:")
        if faiss_matches:
            for i, (response, score) in enumerate(faiss_matches):
                print(f"  Match {i+1}: '{response[:100]}...' (Confidence: {score:.2f}%)")
        else:
            print("  No FAISS matches found or FAISS not loaded.")

        # Get the final RAG system response
        final_response, source, confidence = response_gen.get_response(question)
        print(f"Final RAG System Response (Source: {source}):")
        print(f"  Response: {final_response[:200]}...") # Show more characters for LLM response
        print(f"  Overall Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    test_qa_system()
