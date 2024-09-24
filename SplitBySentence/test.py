import whisper
from pydub import AudioSegment
import nltk
import os

# Download the Punkt tokenizer for sentence splitting (if you haven't already)
nltk.download('punkt')
nltk.download('punkt_tab')

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the audio file with word-level timestamps
result = model.transcribe("SplitBySentence/Unit_1.mp3", word_timestamps=True)


# Create an output directory for sentence audio files
output_dir = "output_sentences"
os.makedirs(output_dir, exist_ok=True)

# Load the original audio file
audio = AudioSegment.from_mp3("SplitBySentence/Unit_1.mp3")

# Function to split audio based on start and end times
def split_audio(audio, start_time, end_time, output_filename):
    start_ms = float(start_time) * 1000  # Convert to milliseconds and ensure proper float type
    end_ms = float(end_time) * 1000
    segment = audio[start_ms:end_ms]
    segment.export(output_filename, format="wav")

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


# Split the audio based on the sentence segments
for idx, sentence in enumerate(sentence_segments):
    output_filename = os.path.join(output_dir, f"{idx}-{sentence['text']}wav")
    split_audio(audio, sentence['start'], sentence['end'], output_filename)
    print(f"Saved: {output_filename} | Text: {sentence['text']}")
