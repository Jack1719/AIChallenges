import whisper
from pydub import AudioSegment
import nltk
import os
import torch
import asyncio
import re

# Download the Punkt tokenizer for sentence splitting (if you haven't already)
nltk.download('punkt')

# Load Whisper model with GPU support if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
print(f"Running on: {device}")

# Transcribe the audio file with word-level timestamps
result = model.transcribe("Unit_1.mp3", word_timestamps=True)

# Create an output directory for sentence audio files
output_dir = "output_sentences"
os.makedirs(output_dir, exist_ok=True)

# Load the original audio file
audio = AudioSegment.from_mp3("Unit_1.mp3")

# Function to split audio based on start and end times
async def async_split_audio(audio, start_time, end_time, output_filename):
    start_ms = float(start_time) * 1000  # Convert to milliseconds
    end_ms = float(end_time) * 1000
    segment = audio[start_ms:end_ms]
    segment.export(output_filename, format="wav")
    print(f"Saved: {output_filename}")

# Function to clean filenames by removing special characters
def clean_filename(text):
    # Replace special characters with underscores
    return re.sub(r'[\\/*?:"<>|]', "_", text)

# Use NLTK to split the full transcription into sentences
sentences = nltk.sent_tokenize(result["text"])

# Map sentences back to word timestamps
sentence_segments = []
current_sentence = ""
start_time = None
sentence_index = 0  # Index to track current sentence
sentence = sentences[sentence_index].strip()

# Process word-level timestamps to match sentences
for segment in result['segments']:
    for word in segment['words']:
        # If we are processing the first word of a sentence, set start_time
        if start_time is None:
            start_time = word['start']

        # Add the current word to the accumulated sentence
        current_sentence += word['word']

        # Check if the accumulated words match the current sentence
        if current_sentence.strip() == sentence:
            end_time = word['end'] + 0.5  # Add padding to the end

            # Save the sentence with its start and end times
            sentence_segments.append({
                'text': current_sentence.strip(),
                'start': start_time,
                'end': end_time
            })

            # Print the matched sentence for debugging purposes
            print(f"Matched: {current_sentence.strip()}")

            # Move to the next sentence
            sentence_index += 1
            current_sentence = ""
            start_time = None

            # Check if all sentences are processed
            if sentence_index >= len(sentences):
                break
            sentence = sentences[sentence_index].strip()

# Asynchronous function to split the audio in reverse order
async def process_audio():
    for idx, sentence in enumerate(reversed(sentence_segments)):
        filename_safe_text = clean_filename(sentence['text'])  # Clean filename
        output_filename = os.path.join(output_dir, f"{100 - idx}-{filename_safe_text}wav")
        await async_split_audio(audio, sentence['start'], sentence['end'], output_filename)

# Run the asynchronous processing
if __name__ == "__main__":
    asyncio.run(process_audio())
