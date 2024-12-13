import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pydub import AudioSegment
from audiorecorder import audiorecorder
import io, numpy as np
import wave
import os

st.set_page_config(layout="wide")

# Set device to mps if available (for Apple Silicon devices)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize Whisper and Llama models
@st.cache_resource
def load_models():
    # huggingface_token = os.getenv("HF_TOKEN", "")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    
    # Initialize the Llama pipeline for text generation with MPS device
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.bfloat16,  # Use bfloat16 if MPS supports it
    )
    
    # Initialize the TTS model
    tts_model = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    
    return processor, model, pipe, tts_model

processor, model, pipe, tts_model = load_models()

# Transcribe audio to text
def transcribe_audio(audio_bytes):
    # Decode audio bytes using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Export to a raw PCM format for librosa
    audio = audio.set_frame_rate(16000).set_channels(1)
    raw_audio = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize PCM values

    # Prepare inputs for Whisper model
    inputs = processor(raw_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Generate response with Llama model
def generate_response(transcription):
    # Prepare conversation for Llama model
    conversation = [
        {"role": "system", "content": "You are a helpful assistant. Summarize your response in 80-100 words"},
        {"role": "user", "content": transcription},
    ]
    
    # Generate response from Llama model
    outputs = pipe(
        conversation,
        max_new_tokens=150,
    )
    
    # Extract only the generated text (not the entire conversation)
    response_text = outputs[0]["generated_text"][-1]
    
    # Return the generated response text
    return response_text["content"]

# Convert text to speech using TTS model and directly prepare for Streamlit playback
def text_to_speech(response):
    # Synthesize speech from text
    speech = tts_model(response)
    
    # speech["audio"] is a numpy array, directly return it
    audio_array = speech["audio"]
    sample_rate = speech["sampling_rate"]  # Get the sample rate directly

    # Ensure the audio is in the correct range for 16-bit audio
    audio_array = np.clip(audio_array, -1.0, 1.0)  # Clip to -1 to 1 range
    audio_array = (audio_array * 32767).astype(np.int16)  # Scale to 16-bit PCM

    # Convert numpy array to WAV format bytes for Streamlit
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())  # Write as 16-bit PCM

    buffer.seek(0)
    return buffer.read()


# Streamlit Interface
st.title("Voice Bot (English)")

audio = audiorecorder(start_prompt="", stop_prompt="")

if len(audio) > 0:
    # To play recorded audio in frontend
    st.audio(audio.export().read(), format="audio/wav")

    # Automatically transcribe after recording stops
    st.write("Transcribing the recorded audio...")
    audio_bytes = audio.export().read()  # Get audio bytes
    transcription = transcribe_audio(audio_bytes)

    # Highlight user prompt
    st.markdown("### User Prompt:")
    st.markdown(f"##### {transcription}")

    # Automatically generate response from Llama model
    st.write("Generating the response...")
    response = generate_response(transcription)
    
    # Highlight AI assistant's latest response only (raw text, not a list of dicts)
    st.markdown("### AI Assistant:")
    st.markdown(f"##### {response}")

    # Convert response to speech and play it
    st.write("Generating speech from response...")
    audio_response = text_to_speech(response)
    
    # Play the TTS audio response
    st.audio(audio_response, format="audio/wav")