import os
import pandas as pd
from modules.utils import Config
from modules.response_gen_rag import ResponseGeneratorRAG

def run_inference(csv_file=None):
    """Run inference on provided test CSV file and update it in place with responses."""
    config = Config()

    # Always use the Data/transcriptions.csv file in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcriptions_csv = os.path.join(project_root, "Data", "transcriptions.csv")
    test_csv_path = csv_file if csv_file is not None else transcriptions_csv

    # Define paths for FAISS artifacts
    faiss_index_path = os.path.join(project_root, "Data", "qa_faiss_index.bin")
    embeddings_file_path = os.path.join(project_root, "Data", "qa_embeddings.npy")

    # Initialize RAG response generator
    response_gen = ResponseGeneratorRAG(
        faiss_index_path=faiss_index_path,
        embeddings_file_path=embeddings_file_path
    )

    try:
        test_df = pd.read_csv(test_csv_path)
        if 'Questions' not in test_df.columns:
            raise ValueError("CSV file must contain a 'Questions' column")
    except Exception as e:
        error_msg = f"Error reading CSV file: {e}"
        print(error_msg)
        return None, error_msg, pd.DataFrame()

    results = []
    for idx, row in test_df.iterrows():
        question = row['Questions']
        print(f"Processing question {idx + 1}/{len(test_df)}: {question[:50]}...")
        response, source, confidence = response_gen.get_response(question)
        results.append({
            'Questions': question,
            'Response': response,
            'Source': source,
            'Confidence': confidence
        })

    output_df = pd.DataFrame(results)
    # Overwrite the original transcriptions.csv file
    output_df.to_csv(transcriptions_csv, index=False, encoding='utf-8-sig')
    print(f"Responses saved to: {transcriptions_csv}")
    return transcriptions_csv

if __name__ == '__main__':
    run_inference()
