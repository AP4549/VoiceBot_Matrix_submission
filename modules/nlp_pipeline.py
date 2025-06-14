import os
import pandas as pd
import json
from typing import Optional, Tuple
from pathlib import Path
from modules.utils import Config, clean_text
from modules.response_gen_rag import ResponseGeneratorRAG
from langdetect import detect

class NLPPipeline:
    def __init__(self):
        self.config = Config()
        
        # Get project root and set up paths
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.faiss_index_path = self.project_root / "Data" / "qa_faiss_index.bin"
        self.embeddings_file_path = self.project_root / "Data" / "qa_embeddings.npy"
        
        # Initialize RAG response generator
        self.response_gen = ResponseGeneratorRAG(
            faiss_index_path=str(self.faiss_index_path),
            embeddings_file_path=str(self.embeddings_file_path)
        )
        
    def detect_language(self, text: str) -> str:
        """Detect language of input text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails
            
    def process_input(self, text: str) -> Tuple[str, str, float]:
        """Process input text and generate response."""
        # Clean and normalize text
        cleaned_text = clean_text(text)
        
        # Get response using RAG
        response, source, confidence = self.response_gen.get_response(cleaned_text)
        
        return response, source, confidence

def run_inference(csv_file=None):
    """Run inference on provided test CSV file and generate responses."""
    config = Config()
    nlp = NLPPipeline()

    # Set up paths
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = project_root / "Data" / "userinputvoice" / "user_inputs.csv"
    output_path = project_root / "Data" / "voicebotoutput" / "voice_responses.csv"
    
    # Use provided CSV or default input path
    test_csv_path = csv_file if csv_file is not None else input_path

    # Read and process input data
    try:
        # Read input CSV
        test_df = pd.read_csv(test_csv_path)
        if 'Question' not in test_df.columns:
            raise ValueError("CSV file must contain a 'Question' column")
    except Exception as e:
        error_msg = f"Error reading CSV file: {e}"
        print(error_msg)
        return None, error_msg, pd.DataFrame()

    results = []
    for idx, row in test_df.iterrows():
        question = row['Question']
        print(f"Processing question {idx + 1}/{len(test_df)}: {question[:50]}...")
        
        # Process through NLP pipeline
        response, source, confidence = nlp.process_input(question)
        
        results.append({
            'Question': question,
            'Response': response,
            'Source': source,
            'Confidence': confidence
        })
        
    # Create output dataframe
    output_df = pd.DataFrame(results)
    
    # Save results to output CSV
    output_df.to_csv(output_path, index=False)
    
    # Generate summary
    total_questions = len(output_df)
    avg_confidence = output_df['Confidence'].mean()
    summary = f"""
    Processing complete.
    Total questions processed: {total_questions}
    Average confidence score: {avg_confidence:.2f}%
    Results saved to: {output_path}
    """
    print(summary)
    return str(output_path), summary, output_df.head(10)

if __name__ == '__main__':
    run_inference()
