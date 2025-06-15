"""
Example test pipeline for Chat Inference and context memory
Run this script after `main.py` dependencies are installed.
"""
from main import VoiceBotUI


def test_pipeline():
    # Initialize VoiceBot (loads AWS and Supabase clients)
    vb = VoiceBotUI()

    # Sign in with a test account
    email = "poker.ap4549@gmail.com"
    password = "VoiceBot@2025!"
    success, auth_msg = vb.sign_in_user(email, password)
    print(f"Sign in: {auth_msg}")
    if not success:
        return

    # In-memory history for text chat
    history = []

    # First question: credit score
    q1 = "What is credit score?"
    r1 = vb.process_text_input(q1, history)
    history.append([q1, r1])
    print(f"User: {q1}\nBot: {r1}\n")

    # Second question: docs for same topic
    q2 = "And for the same topic, what documents are required?"
    r2 = vb.process_text_input(q2, history)
    history.append([q2, r2])
    print(f"User: {q2}\nBot: {r2}\n")

    # Third question: ask about previous question (fallback)
    q3 = "What was my previous question?"
    r3 = vb.process_text_input(q3, history)
    print(f"User: {q3}\nBot: {r3}\n")


if __name__ == "__main__":
    test_pipeline()
