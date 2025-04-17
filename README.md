import streamlit as st
import speech_recognition as sr
from llama_cpp import Llama

# Load LLaMA 2 model once
from rom_llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="TheBloke/Llama-2-7B-GGUF",
	filename="llama-2-7b.Q2_K.gguf",
)


# Title
st.title("🎤 Speech to LLaMA Chatbot")
st.markdown("Click the button and start speaking...")

# UI button to trigger recording
if st.button("🎙️ Record and Ask"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Speech to text
        prompt = recognizer.recognize_google(audio)
        st.write("🗣️ You said:", prompt)

        # Get LLaMA's response
        output = llm(prompt=prompt, max_tokens=100, stop=["</s>"])
        response = output["choices"][0]["text"].strip()

        # Show the result
        st.success("🤖 LLaMA says:")
        st.write(response)

    except sr.UnknownValueError:
        st.error("Could not understand your voice.")
    except sr.RequestError as e:
        st.error(f"Google Speech Recognition error: {e}")
