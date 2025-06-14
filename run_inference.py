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
        return None, error_msg, pd.DataFrame()

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
    
    print(summary)
    
    # Prepare preview of first 10 questions for Gradio
    preview_df = test_df['Questions'].head(10).to_frame()
    
    return output_path, summary, preview_df

def create_gradio_interface():
    """Create and launch the Gradio interface with the current color theme and added hover effects for buttons and tabs. Only the batch process tab is shown."""
    try:
        custom_css = """
        .gr-button, .gradio-button, button, .gradio-file label, .gradio-file input[type='file']::file-selector-button {
            transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
        }
        .gr-button:hover, .gradio-button:hover, button:hover {
            filter: brightness(0.95) contrast(1.1);
            box-shadow: 0 4px 16px rgba(28,181,224,0.18);
            transform: translateY(-2px) scale(1.03);
        }
        .gradio-tabs .tabitem, .gradio-tab {
            transition: background 0.2s, color 0.2s;
        }
        .gradio-tabs .tabitem:hover, .gradio-tab:hover {
            filter: brightness(1.08) contrast(1.1);
            box-shadow: 0 2px 8px rgba(28,181,224,0.10);
        }
        /* Custom font for CSV preview (using Times New Roman) */
        .csv-preview {
            font-family: 'Times New Roman', serif;
        }
        """
        with gr.Blocks(title="VoiceBot MATRIX - RAG QA System", css=custom_css) as demo:
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
                gr.Markdown("## CSV Preview (First 10 Rows)")
                csv_preview = gr.Dataframe(headers=["Questions"], type="pandas", row_count=10, col_count=1, interactive=False, elem_id="csv_preview", elem_classes=["csv-preview"])

            # Set up event handler only for batch process
            process_btn.click(run_inference, inputs=[csv_input], outputs=[output_file, result_text, csv_preview])

        demo.launch(share=False)

    except Exception as e:
        print(f"Error starting Gradio interface: {e}")
        print("Falling back to command-line interface...")
        # Optionally, you can remove the CLI fallback for single question as well, since interactive QA is being removed.

if __name__ == '__main__':
    create_gradio_interface()