import gradio as gr
import threading
import numpy as np

# Import your existing components
from modules.main import SpeechRecognizer, ResponseGenerator, VoiceSynthesizer

# Initialize components once
recognizer = SpeechRecognizer()
response_gen = ResponseGenerator() 
tts = VoiceSynthesizer()

# Global conversation history
chat_history = []

def process_audio(audio):
    """Handle audio input from Gradio mic"""
    global chat_history
    
    if audio is None:
        return chat_history, None
    
    # audio comes as (sample_rate, audio_array) from Gradio
    sample_rate, audio_array = audio
    audio_array = audio_array.astype(np.int16)
    
    try:
        # Step 1: Transcribe
        question = recognizer.transcribe(audio_array, sample_rate)
        if not question.strip():
            return chat_history, None
            
        # Add to chat history
        chat_history.append(("You", question))
        
        # Step 2: Generate response
        response = response_gen.get_response(question)
        chat_history.append(("AI", response))
        
        # Step 3: Speak in background thread
        def speak():
            tts.speak(response)
        threading.Thread(target=speak).start()
        
    except Exception as e:
        chat_history.append(("System", f"Error: {str(e)}"))
    
    return chat_history, None  # Clear mic input

# Create Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="VoiceBot MATRIX") as demo:
    gr.Markdown("# 🎙️ VoiceBot MATRIX")
    
    with gr.Row():
        with gr.Column(scale=2):
            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Press & hold to speak",
                show_label=True,
                interactive=True
            )
            clear_btn = gr.Button("Clear Chat")
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation History",
                bubble_full_width=False,
                show_label=True
            )
    
    # Instructions
    gr.Markdown("### Instructions")
    gr.Markdown("1. Press and hold the microphone button to speak\n"
                "2. Release when finished\n"
                "3. Wait for the AI response (audio will play automatically)")
    
    # Connect components
    mic.change(
        fn=process_audio,
        inputs=mic,
        outputs=[chatbot, mic],
        show_progress="hidden"
    )
    
    clear_btn.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False
    )

if __name__ == "__main__":
    demo.launch(show_api=False)
