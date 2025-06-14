import sounddevice as sd
import wavio

def record_audio(filename='user_input.wav', duration=5, fs=16000):
    """
    Records audio from the microphone and saves it as a WAV file.

    Args:
        filename (str): Output WAV file name.
        duration (int): Duration of recording in seconds.
        fs (int): Sampling rate (Hz).
    """
    print("Recording... Please speak into your microphone.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"Recording finished. Audio saved as {filename}")

# Example usage:
record_audio('user_input_1.wav', duration=5)
