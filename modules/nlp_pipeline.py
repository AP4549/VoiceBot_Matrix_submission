import os
import pandas as pd
import json
import time
from typing import Optional, Tuple, Dict, List, Generator, Any
from pathlib import Path
from modules.utils import Config, clean_text
from VoiceBot_MATRIX_submission.modules.response_gen_ragb import ResponseGeneratorRAG
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        # Configure chunk sizes and processing parameters
        self.chunk_size = 5  # Process 5 items at a time
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def detect_language(self, text: str) -> Tuple[str, bool]:
        """Detect language of input text and check if it's Hinglish."""
        try:
            # Check for Devanagari characters first
            devanagari_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')
            if devanagari_count > 0:
                return 'hi', False

            # Check for Hinglish
            is_hinglish = self._is_hinglish(text)
            if is_hinglish:
                return 'hi', True

            # Use langdetect for other cases
            lang = detect(text)
            if lang not in ['en', 'hi']:
                # Check for common Hindi/Hinglish words
                hindi_words = ['hai', 'kya', 'main', 'nahi', 'aap', 'kaise', 'mujhe', 'hum', 'tum', 'kar', 'koi']
                text_lower = text.lower()
                if any(word in text_lower.split() for word in hindi_words):
                    return 'hi', True
            return lang if lang in ['en', 'hi'] else 'en', False
        except:
            return 'en', False

    def _is_hinglish(self, text: str) -> bool:
        """Check if the text is Hinglish (Hindi written in Roman script)"""
        if not text:
            return False
        try:
            devanagari_count = sum('\u0900' <= c <= '\u097F' for c in text)
            latin_count = sum('a' <= c.lower() <= 'z' for c in text if c.isalpha())
            return devanagari_count == 0 and latin_count > 3
        except:
            return False

    def process_with_retry(self, text: str) -> Tuple[str, str, float]:
        """Process input with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                cleaned_text = clean_text(text)
                response, source, confidence = self.response_gen.get_response(cleaned_text)
                return response, source, confidence
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return "An error occurred while processing your request.", "error", 0.0
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        return "System is currently unavailable.", "error", 0.0

    def process_input(self, text: str) -> Tuple[str, str, float]:
        """Process single input text and generate response."""
        return self.process_with_retry(text)

    def process_chunk(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process a chunk of questions with proper error handling."""
        results = []
        for question in questions:
            try:
                response, source, confidence = self.process_with_retry(question)
                results.append({
                    'Question': question,
                    'Response': response,
                    'Source': source,
                    'Confidence': confidence
                })
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                results.append({
                    'Question': question,
                    'Response': "Error processing this question.",
                    'Source': "error",
                    'Confidence': 0.0
                })
        return results

    def process_batch(self, questions: List[str]) -> Generator[Dict[str, Any], None, None]:
        """Process questions in chunks and yield results as they become available."""
        for i in range(0, len(questions), self.chunk_size):
            chunk = questions[i:i + self.chunk_size]
            try:
                results = self.process_chunk(chunk)
                for result in results:
                    yield result
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                for question in chunk:
                    yield {
                        'Question': question,
                        'Response': "Error processing this question.",
                        'Source': "error",
                        'Confidence': 0.0
                    }

def run_inference(csv_file=None) -> Tuple[Optional[str], str, pd.DataFrame]:
    """Run inference on provided test CSV file with improved error handling."""
    config = Config()
    nlp = NLPPipeline()

    # Set up paths
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = project_root / "Data" / "userinputvoice" / "user_inputs.csv"
    output_path = project_root / "output" / "output.csv"
    
    # Use provided CSV or default input path
    test_csv_path = csv_file if csv_file is not None else input_path

    try:
        # Read input CSV with proper encoding handling
        test_df = pd.read_csv(test_csv_path, encoding='utf-8-sig')
        question_col = next((col for col in ['Questions', 'Question'] if col in test_df.columns), None)
        if not question_col:
            raise ValueError("CSV file must contain a 'Questions' or 'Question' column")

        print(f"Processing {len(test_df)} questions...")
        results = []
        
        # Process questions and collect results
        for result in nlp.process_batch(test_df[question_col].tolist()):
            results.append(result)
            
        # Create output dataframe
        output_df = pd.DataFrame(results)
        
        # Ensure output directory exists
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Save results in chunks to avoid memory issues
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig', chunksize=1000)
        
        # Generate summary
        processed = len(output_df)
        successful = sum(1 for r in results if r['Source'] != "error")
        errors = processed - successful
        avg_confidence = output_df[output_df['Source'] != "error"]['Confidence'].mean()
        
        summary = f"""Processing Summary:
        Total questions: {processed}
        Successfully processed: {successful}
        Errors: {errors}
        Average confidence: {avg_confidence:.2f}%
        Results saved to: {output_path}
        """
        
        print(summary)
        return str(output_path), summary, output_df.head(10)

    except Exception as e:
        error_msg = f"Error during inference: {str(e)}"
        print(error_msg)
        return None, error_msg, pd.DataFrame()

if __name__ == '__main__':
    run_inference()
