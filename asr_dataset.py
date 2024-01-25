import os
import whisper
import json

# def transcribe_audio(audio_seg):
#     # using whisper model to transcribe and return list[text], list[text, [{word, start_time, end_time}]]
#     model = whisper.load_model(model_size)
#     result = model.transcribe("audio1.mp3", word_timestamps=True)
#     for e in result["segments"]:
#     print("\n", e["text"], "\n", e["words"]) # each contains a list of dictionaries [{'word':'xyz', 'start':0.0, 'end':0.45,..}, {...}, {...}]
"""
ffmpeg is required

sudo apt install ffmpeg
"""

class audio_transcriber:

    def __init__(self, model_size="base", word_timestamp=False):
        self.timestamp = word_timestamp
        self.model_size = model_size
        self.model = whisper.load_model(self.model_size)

    def __call__(self, audio_seg):
        result = self.model.transcribe(audio_seg, word_timestamps=self.timestamp)
        # each "segments" contains words spoken together, thus each "text" is a sentence/group of words
        # each "words" contains a list of dictionaries [{'word':'xyz', 'start':0.0, 'end':0.45,..}, {...}, {...}]
        words_timestamp = [[e["text"], e["words"]] for e in result["segments"]]
        return result["text"], words_timestamp 


def traverse_directories(base_dir, result_dir, asr_model):
    list_dir = os.listdir(base_dir)

    for dir in list_dir:
        text, words_timestamp = asr_model(os.path.join(base_dir, dir, "audio.wav"))
        
        # save complete transcript of audio
        with open(os.path.join(result_dir, dir+'.txt'), "w+") as f:
            f.write(text)

        # save python list containing individual sentences and time stamp of each word of the sentence
        with open(os.path.join(result_dir, dir+'.json'), "w+") as f:
            json.dump(words_timestamp, f)



if __name__ == '__main__':
    # setup directories
    base_dir = "/home/ubuntu/audio_data"
    result_dir = "/home/ubuntu/audio_text"
    os.makedirs(result_dir, exist_ok=True)

    # model
    asr_model = audio_transcriber("tiny", True)

    # perform transcription
    traverse_directories(base_dir, result_dir, asr_model)
