import os
import cv2
import re

def natural_sort_key(s):
    """Function to define natural sorting key."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_video_from_frames(input_dir, output_file, fps=30):
    # Get list of frame filenames
    frame_files = sorted(os.listdir(input_dir), key=natural_sort_key)

    # Get frame size from first frame
    first_frame = cv2.imread(os.path.join(input_dir, frame_files[0]))
    frame_height, frame_width, _ = first_frame.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Iterate through frames and write to video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(input_dir, frame_file))
        out.write(frame)

    # Release VideoWriter
    out.release()

    print(f"Video created successfully: {output_file}")

# Example usage
input_directory = '....'
output_video = 'output_video.mp4'
create_video_from_frames(input_directory, output_video)
