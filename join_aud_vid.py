import subprocess

def join_audio_to_video(video_file, audio_file, output_file):
    command = [
        'ffmpeg',
        '-i', video_file,
        '-i', audio_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-map', '0:v:0',
        '-map', '1:a:0',
        output_file
    ]

    subprocess.run(command, check=True)

    print(f"Audio added to video successfully: {output_file}")

# Example usage
video_file = 'output_video.mp4'
audio_file = 'audio_file.mp3'
output_file = 'video_with_audio.mp4'

join_audio_to_video(video_file, audio_file, output_file)
