import os
import pandas as pd
from modules.response_gen import ResponseGenerator
from modules.utils import Config

def run_inference():
    """Run inference on test.csv and generate output.csv"""
    config = Config()
    
    # Initialize response generator
    response_gen = ResponseGenerator()
    
    # Read test.csv
    test_csv_path = os.path.join(os.path.dirname(__file__), config.get('test_csv'))
    try:
        test_df = pd.read_csv(test_csv_path)
        if 'Questions' not in test_df.columns:
            raise ValueError("test.csv must contain a 'Questions' column")
    except Exception as e:
        print(f"Error reading test.csv: {e}")
        return

    # Process each question
    results = []
    for idx, row in test_df.iterrows():
        question = row['Questions']
        print(f"Processing question {idx + 1}/{len(test_df)}: {question[:50]}...")
        
        response, source, confidence = response_gen.get_response(question)
        results.append({
            'Question': question,
            'Response': response,
            'Source': source,
            'Confidence': confidence
        })
        
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save output
    output_path = os.path.join(os.path.dirname(__file__), config.get('output_csv'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Processed {len(results)} questions")
    print(f"Dataset matches: {sum(1 for r in results if r['Source'] == 'dataset')}")
    print(f"LLM fallbacks: {sum(1 for r in results if r['Source'] == 'llm')}")
    print(f"Average confidence: {sum(r['Confidence'] for r in results) / len(results):.2f}%")

if __name__ == '__main__':
    run_inference()
