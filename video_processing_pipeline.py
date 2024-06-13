import os
import json
import subprocess
import shutil
## import func to extr_video
from ASR_NER.extract_frames import extract_frames
from ASR_NER.ext_aud import extract_audio
from ASR_NER.asr_dataset import audio_transcriber
from ASR_NER.ner_dataset import NER_model, partial_match
from ASR_NER.mute_audio import mute_segments
from ASR_NER.create_video import create_video_from_frames
from ASR_NER.join_aud_vid import join_audio_to_video
import argparse
import random

def extract_pii(base_dir, result_dir, pii_model, alw_tag):
    print("\n", "Tagging ...", base_dir)
    # filename = file.split('/')[-1].split('.')[0]
    filename = 'text_pii'

    # open json file for segments and timings
    with open(base_dir+'sent_timestamp.json') as f:
        data = json.load(f)

    # store start and end timestamp for whole transcript
    redact_time = [] 
    tagged_text = []

    # iterating over each segment of the transcript
    for segment in data:
    
        res = pii_model(segment[0]) # passing text of the whole transcript
        tagged_text.append(res)
        
        # iter over tags found in the seg
        for wrd_tags in res[1:]:

            # ignore tags not in list
            if wrd_tags['tag'] not in alw_tag:
                continue

            # iter over words in seg - to find the word's start and end
            for idx in range(len(segment[1])):
                if partial_match(wrd_tags['word'], segment[1][idx]['word'], 0.3):
                    # adding tag and word for purpose of debugging
                    redact_time.append((wrd_tags['tag'], segment[1][idx]['word'], segment[1][idx]['start'], segment[1][idx]['end']))
                
    # save python list containing text segments, words and tags
    with open(os.path.join(result_dir, filename +'_tags.json'), "w+") as f:
        json.dump(tagged_text, f)

    # save list of timestamps to mute the pii's
    with open(os.path.join(result_dir, filename +'_redact.json'), "w+") as f:
        json.dump(redact_time, f)            


def compare_and_copy_images(folder_a, folder_b):
    # Get the list of files in both folders
    files_in_a = set(os.listdir(folder_a))
    files_in_b = set(os.listdir(folder_b))
    
    # Identify the missing files in folder B
    missing_files = files_in_a - files_in_b
    
    # Filter only image files (assuming images have standard extensions)
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    missing_images = [f for f in missing_files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Copy missing images from A to B
    for image in missing_images:
        src_path = os.path.join(folder_a, image)
        dest_path = os.path.join(folder_b, image)
        shutil.copy2(src_path, dest_path)
        print(f"Copied {image} to {folder_b}")
    print('checking ----')

    
def ai_pipeline(base_dir, filename):
    # base_dir = "/home/ubuntu/test/"
    # orig_vid_dir = base_dir + 'video.mp4'

    path_vid_filename = os.path.join(base_dir, os.path.join('input', filename))

    # create intermediate folder
    internal_dir = os.path.join(base_dir, "internal/")
    os.makedirs(internal_dir, exist_ok=True)

    # create output folder
    output_dir = os.path.join(base_dir, "output/")
    os.makedirs(output_dir, exist_ok=True)
    
    ## function to extract frames
    frame_dir = os.path.join(internal_dir, 'frames/')
    os.makedirs(frame_dir, exist_ok=True)
    print("... Extracting frames from video ...")
    extract_frames(path_vid_filename, frame_dir) 


    ## Command 1: Running detect.py with sudo and setting CUDA_VISIBLE_DEVICES
    print("\n... Running object detection model ....")
    detect_command = [
        'sudo',
        'CUDA_VISIBLE_DEVICES=0',
        'python3', 
        '/opt/object-detection/yolov5/detect.py', 
        '--source', frame_dir, 
        '--conf', '0.25', 
        '--weights', 'runs/train/train_cliff_sample2/weights/best.pt', 
        '--save-txt',
        '--project', internal_dir + 'object_detect/detection/'
    ]
    # Using subprocess to run the command
    subprocess.run(' '.join(detect_command), shell=True, check=True)
    print("object detection done")

    # Command 2: Running 4_blur_faces.py
    print("\n... Face blur operation started ...")
    blur_command = [
        'python3', 
        '/opt/object-detection/yolov5/4_blur_faces.py', 
        '-i', frame_dir, 
        '-l', internal_dir + 'object_detection/detection/exp/labels/', ##!!
        '-o', internal_dir + 'object_detection/blurred_object_detection/', ##!!
        '-r', '0.9'
    ]
    # Using subprocess to run the command
    subprocess.run(blur_command)
    print("blurring on object detection done")

    ## compare prev frames to blur and copy the remaining
    compare_and_copy_images(frame_dir, internal_dir + 'object_detect/blurred_object_detection/')


    ## extract audio from original video
    audio_dir = internal_dir + 'audio/'
    print("\n... Extracting audio from original video ...")
    os.makedirs(audio_dir, exist_ok=True)
    extract_audio(path_vid_filename, audio_dir)

    
    ## get text from audio
    text_dir = output_dir + 'text/'
    print("\n... Generating transcript from the audio ...")
    os.makedirs(text_dir, exist_ok=True)
    asr_model = audio_transcriber("large", True) # init ASR model ("medium.en" or "large" or "large-v3")
    # run model
    text, words_timestamp = asr_model(os.path.join(audio_dir, "audio.wav"))
    # save transcript
    with open(os.path.join(text_dir, 'transcript.txt'),"w+") as f:
        f.write(text)
    # save python list containing indiv sent and time stamp
    with open(os.path.join(text_dir, 'sent_timestamp.json'),"w+") as f:
        json.dump(words_timestamp, f)
    
    
    # get pii from text
    text_pii_dir = output_dir + 'text_pii/'
    print("\n... Detecting PII from the transcript ...")
    os.makedirs(text_pii_dir, exist_ok=True)
    alw_tag = ["PERSON", "ORG", "LOC"] # ['MISC'], 'GPE', 'FAC', 'EVENT', 'LAW'
    pii_model = NER_model() # 'ner-ontonotes-large'
    extract_pii(text_dir, text_pii_dir, pii_model, alw_tag)
    
    ## beep audio
    filename = 'text_pii'
    beep_aud_dir = output_dir + "beep_aud/"
    print("\n... Adding beep sound to audio based on PII ...")
    os.makedirs(beep_aud_dir, exist_ok=True)
    print("Replacing by beep sound..")
    with open(text_pii_dir+filename+"_redact.json") as f:
        mute_segments_list = json.load(f)
    mute_segments_list.sort(key=lambda x:x[2])
    mute_segments(audio_dir+"audio.wav", mute_segments_list, beep_aud_dir+"beep.wav")

    ## create video from result frames
    input_res_frames = internal_dir + "object_detect/blurred_object_detection/"
    output_video = internal_dir + "output_video.mp4"
    print("\n... Generating video from resultant frames ...")
    create_video_from_frames(input_res_frames, output_video)
    
    ## merge audio and video 
    output_aud_vid = output_dir + "video_with_audio.mp4"
    print("\n... Merging final video and audio as final result ...")
    join_audio_to_video(output_video, beep_aud_dir+"beep.wav", output_aud_vid)

    ## thumbnail
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    thumb_options = [f for f in frame_dir if os.path.splitext(f)[1].lower() in image_extensions]
    thumb_img = random.choice(thumb_options)
    src_path = os.path.join(frame_dir, thumb_img)
    dest_path = os.path.join(output_dir, 'thumbnail.jpg')
    shutil.copy2(src_path, dest_path)
    print(f"Copied {thumb_img} as thumbnail")
    

def parse_args():
    parser = argparse.ArgumentParser()

    ## path to images
    parser.add_argument('--base_dir', '-b',
                        type=str, required=False,
                        help="Path to directory")
    
    ## name of file
    parser.add_argument('--filename', '-f',
                        type=str, required=True,
                        help='name of the video')
    
    args = parser.parse_args()
    return args


def main(args):
    ai_pipeline(args.base_dir, args.filename)


if __name__ == '__main__':
    args = parse_args()
    main(args)