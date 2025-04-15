# Step 1: Install gTTS
!pip install gTTS

# Step 2: Import necessary libraries
from gtts import gTTS
from IPython.display import Audio

# Step 3: Get input from the user
user_text = input("Enter the text you want to convert to speech: ")

# Step 4: Convert text to speech
tts = gTTS(text=user_text, lang='en')  # You can change 'en' to other language codes like 'hi', 'es', etc.
tts.save("output.mp3")

# Step 5: Play the audio
Audio("output.mp3")
