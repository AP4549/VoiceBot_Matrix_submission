import pandas as pd
from modules.response_gen_rag import ResponseGeneratorRAG

# Test questions in both English and Hindi
test_questions = [
    "What is the rate of interest for personal loan?",
]

def test_qa_system():
    print("Testing QA System...")
    faiss_index_path = "Data/qa_faiss_index.bin"
    embeddings_file_path = "Data/qa_embeddings.npy"
    response_gen = ResponseGeneratorRAG(faiss_index_path=faiss_index_path, embeddings_file_path=embeddings_file_path)
    
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
