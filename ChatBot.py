import pyttsx3
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
import webbrowser
from urllib.parse import urlencode
import requests

# Initialize the pyttsx3 engine for text-to-speech (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set the speed of speech
engine.setProperty('volume', 0.9)  # Set the volume (range: 0.0 to 1.0)

# Load a pre-trained model and tokenizer from Hugging Face's DialoGPT
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load the DialoGPT model
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer

# Function to convert speech to text (STT)
def recognize_speech():
    """
    It uses the mic to reccord the user's voice and then turns it into text. 
    Returns the text the recognized text or None if no valid input was detected.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        try:
            # Listen for user input with a timeout and phrase time limit
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Recognizing...")
            # Use Google Web Speech API to recognize speech
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            # Handle cases where the speech is not understood
            return None
        except (sr.RequestError, sr.WaitTimeoutError):
            # Handle errors gracefully
            return None

# Function to generate a response using DialoGPT
def generate_response(user_input):
    """
    Generates a chatbot response to the user's input using DialoGPT.
    Args:
        user_input (str): The input text from the user.
    Returns:
        str: The generated response.
    """
    # Tokenize and encode the user input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Generate a response using the model
    response = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and return the generated response
    return tokenizer.decode(response[:, inputs.shape[-1]:][0], skip_special_tokens=True)

# Function to convert text to speech (TTS)
def speak_text(text):
    """
    Converts the given text to speech and speaks it aloud.
    Args:
        text (str): The text to speak.
    """
    engine.say(text)
    engine.runAndWait()

def play_song(song_name):
    """
    Searches for a song on YouTube and opens the search results in the browser.
    Args:
        song_name (str): The name of the song to search for.
    """
    # Construct the YouTube search URL
    search_url = "https://www.youtube.com/results"
    query_string = urlencode({"search_query": song_name})
    full_url = f"{search_url}?{query_string}" 

    # Send a GET request to YouTube
    response = requests.get(full_url)
    if response.status_code == 200:
        # Open the search results in the default web browser
        webbrowser.open(full_url)
        speak_text(f"Here are the results for {song_name} on YouTube.")
    else:
        speak_text("Sorry, I couldn't play that song. Please try again.")

# Function to open websites based on voice commands
def open_website(command):
    """
    Opens specific websites based on the user's command.
    Args:
        command (str): The command text specifying the website to open.
    """
    if "open youtube" in command:
        speak_text("Opening YouTube")
        webbrowser.open("https://www.youtube.com")
    elif "open google" in command:
        speak_text("Opening Google")
        webbrowser.open("https://www.google.com")
    elif "open github" in command:
        speak_text("Opening GitHub")
        webbrowser.open("https://github.com/Ravi-S-Vishwakarma-source/Spoken-Interface-Project-ChatBot-.git")
    else:
        speak_text("Sorry, I can't open that website.")

# Main loop to run the chatbot
def voice_assistant():
    """
    Runs the voice assistant in a loop, listening for user input and responding.
    """
    print("Chatbot is ready to chat! Say 'exit' to quit.")
    while True:
        # Get user input through speech-to-text
        user_input = recognize_speech()
        if user_input:
            print(f"You said: {user_input}")

            if "play me a song" in user_input.lower():
                # Handle the "play me a song" command
                speak_text("Sure! What song would you like me to play?")
                song_name = recognize_speech()
                if song_name:
                    play_song(song_name)
                else:
                    speak_text("Sorry, I didn't catch that. Please try again.")
                        
            elif "open" in user_input.lower():
                # Handle commands to open websites
                open_website(user_input.lower())
            elif "exit" in user_input.lower():
                # Exit the chatbot loop
                speak_text("Goodbye! Have a great day!")
                break
            else:
                # Generate a chatbot response for general inputs
                response = generate_response(user_input)
                print(f"Chatbot: {response}")
                speak_text(response)
        else:
            print("No valid input detected. Please try speaking again.")

# Entry point for the program
if __name__ == "__main__":
    voice_assistant()
