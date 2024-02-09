import json
import os
from pydub import AudioSegment

'''
pip install pydub
'''

def mute_segments(audio_path, segments, output_path):
    audio = AudioSegment.from_file(audio_path)

    # Convert segment times from seconds to milliseconds
    segments = [(start * 1000-250, end * 1000+150) for _,_,start, end in segments]

    # Iterate over segments
    for start, end in segments:
        # Replace the corresponding segment in audio with silence
        audio = audio[:start] + AudioSegment.silent(duration=end-start) + audio[end:]

    # Export the muted audio to a file
    audio.export(output_path, format="wav")


def traverse_directories(base_dir, result_dir, segment_dir):
    list_dir = os.listdir(base_dir)

    for dir in list_dir:
        input_file = os.path.join(base_dir, dir, "audio.wav")
        output_file = os.path.join(result_dir, dir+"_muted.wav")
        segment_path = os.path.join(segment_dir, dir+"_redact.json")

        print("Muting audio...", dir)

        with open(segment_path) as f:
            mute_segments_list = json.load(f)

        mute_segments_list.sort(key = lambda x: x[2])

        mute_segments(input_file, mute_segments_list, output_file)


if __name__ == '__main__':
    # setup directories
    base_dir = "/home/ubuntu/prev_data" # audio_data
    result_dir = "/home/ubuntu/prev_muted" # audio_muted
    os.makedirs(result_dir, exist_ok=True)
    segment_dir = "/home/ubuntu/prev_text_pii" # text_pii

    traverse_directories(base_dir, result_dir, segment_dir)
