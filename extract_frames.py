import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5)

    print(f"Video resolution: {frame_width} x {frame_height}, FPS: {fps}")

    # Iterate through each frame and save it
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame
        # frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        # cv2.imwrite(frame_filename, frame)
        # Save the frame as JPEG with quality set to 95 (adjust as needed)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        frame_count += 1

        # Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    # Release the video capture object
    cap.release()
    print(f"Frame extraction complete for {video_path}.")

if __name__ == "__main__":
    # Replace 'input_videos_folder' with the path to the folder containing video files
    input_videos_folder = '/home/ai-team/Desktop/DATASET/Audio_data/cliff42'

    # Iterate through each video file in the input folder
    for video_file in os.listdir(input_videos_folder):
        video_path = os.path.join(input_videos_folder, video_file)

        # Ensure the file is a video file (you can customize this check based on your video file extensions)
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            # Replace 'output_frames_folder' with the name of the folder where you want to save frames
            output_frames_folder = os.path.join('/home/ai-team/Desktop/Frames', os.path.splitext(video_file)[0])

            extract_frames(video_path, output_frames_folder)
