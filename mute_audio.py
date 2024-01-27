import json
from pydub import AudioSegment

from pydub import AudioSegment

def mute_segments(audio_path, segments, output_path):
    audio = AudioSegment.from_file(audio_path)

    # Convert segment times from seconds to milliseconds
    segments = [(start * 1000-200, end * 1000+100) for _,_,start, end in segments]

    # Iterate over segments
    for start, end in segments:
        # Replace the corresponding segment in audio with silence
        audio = audio[:start] + AudioSegment.silent(duration=end-start) + audio[end:]

    # Export the muted audio to a file
    audio.export(output_path, format="wav")


if __name__ == '__main__':
    # Example usage
    input_file = "../audio/interview/audio.wav"
    output_file = "../audio_muted/interview_muted.wav"
    segment_path = '../text_pii/interview_redact.json'

    with open(segment_path) as f:
        mute_segments_list = json.load(f)

    mute_segments_list.sort(key = lambda x: x[2])
    # mute_segments_list = [(5, 10), (20, 25)]  # List of tuples representing start and end times in seconds

    mute_segments(input_file, mute_segments_list, output_file)
