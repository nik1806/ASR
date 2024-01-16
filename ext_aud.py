from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def extract_audio(video_path, output_audio_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_audio_folder):
        os.makedirs(output_audio_folder)

    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Extract audio
    audio_clip = video_clip.audio

    # Save the audio file
    audio_filename = os.path.join(output_audio_folder, f"audio.wav")
    audio_clip.write_audiofile(audio_filename)

    # Close the video clip
    video_clip.close()

    print(f"Audio extraction complete for {video_path}.")

if __name__ == "__main__":
    # Replace 'input_videos_folder' with the path to the folder containing video files
    input_videos_folder = 'input_videos_folder'

    # Iterate through each video file in the input folder
    for video_file in os.listdir(input_videos_folder):
        video_path = os.path.join(input_videos_folder, video_file)

        # Ensure the file is a video file (you can customize this check based on your video file extensions)
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            # Replace 'output_audio_folder' with the name of the folder where you want to save audio
            output_audio_folder = os.path.join('output_audio_folder', os.path.splitext(video_file)[0])

            extract_audio(video_path, output_audio_folder)
