import whisper

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe("Unit_1.mp3", word_timestamps=True)

# result['text'] contains the transcription
# result['segments'] contains timestamps
