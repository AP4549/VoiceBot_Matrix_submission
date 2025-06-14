import os
import pandas as pd
import gradio as gr
from modules.response_gen_rag import ResponseGeneratorRAG
from modules.utils import Config

def run_inference(csv_file=None):
    """Run inference on provided test CSV file and generate output.csv"""
    config = Config()
    
    # Define paths for FAISS artifacts
    faiss_index_path = "Data/qa_faiss_index.bin"
    embeddings_file_path = "Data/qa_embeddings.npy"
    
    # Initialize RAG response generator
    response_gen = ResponseGeneratorRAG(
        faiss_index_path=faiss_index_path, 
        embeddings_file_path=embeddings_file_path
    )
    
    # Use provided CSV or default
    test_csv_path = csv_file if csv_file is not None else os.path.join(os.path.dirname(__file__), config.get('test_csv'))
    
    try:
        test_df = pd.read_csv(test_csv_path)
        if 'Questions' not in test_df.columns:
            raise ValueError("Test CSV file must contain a 'Questions' column")
    except Exception as e:
        error_msg = f"Error reading test CSV file: {e}"
        print(error_msg)
        return None, error_msg

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
         
    output_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), config.get('output_csv'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    summary = f"\nResults saved to {output_path}\n"
    summary += f"Processed {len(results)} questions\n"
    summary += f"Dataset matches: {sum(1 for r in results if 'dataset' in r['Source'])}\n"
    summary += f"LLM fallbacks: {sum(1 for r in results if 'llm' in r['Source'])}\n"
    summary += f"Average confidence: {sum(r['Confidence'] for r in results) / len(results):.2f}%"
    
    print(summary)
    return output_path, summary

def process_single_question(question):
    """Process a single question for the interactive Gradio interface"""
    faiss_index_path = "Data/qa_faiss_index.bin"
    embeddings_file_path = "Data/qa_embeddings.npy"
    response_gen = ResponseGeneratorRAG(faiss_index_path=faiss_index_path, embeddings_file_path=embeddings_file_path)
    response, source, confidence = response_gen.get_response(question)
    return f"Response: {response}\n\nSource: {source}\nConfidence: {confidence:.2f}%"

def create_gradio_interface():
    """Create and launch the Gradio interface"""
    try:
        with gr.Blocks(title="VoiceBot MATRIX - RAG QA System") as demo:
            gr.Markdown("# Voice Bot MATRIX RAG QA System")

            with gr.Tab("Batch Process"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Upload a CSV file with questions")
                        gr.Markdown("The CSV must have a column named 'Questions'")
                        csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                        process_btn = gr.Button("Process Questions")
                    with gr.Column():
                        output_file = gr.File(label="Output CSV")
                        result_text = gr.Textbox(label="Results Summary", lines=6)

            with gr.Tab("Interactive QA"):
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...")
                        ask_btn = gr.Button("Get Answer")
                    with gr.Column():
                        answer_output = gr.Textbox(label="Answer", lines=10)

            # Event handlers
            process_btn.click(run_inference, inputs=[csv_input], outputs=[output_file, result_text])
            ask_btn.click(process_single_question, inputs=[question_input], outputs=[answer_output])

        demo.launch(share=False)

    except Exception as e:
        print(f"Error starting Gradio interface: {e}")
        print("Falling back to command-line interface...")
        while True:
            try:
                question = input("\nEnter your question (or 'exit' to quit): ")
                if question.lower() in ['exit', 'quit']:
                    break
                answer = process_single_question(question)
                print("\n" + answer)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing question: {e}")

if __name__ == '__main__':
    create_gradio_interface()
