import os
import json
import subprocess
## import func to extr_video
from extract_frames import extract_frames
from ext_aud import extract_audio
from asr_dataset import audio_transcriber
from ner_dataset import NER_model, partial_match
from mute_audio import mute_segments
from create_video import create_video_from_frames
from join_aud_vid import join_audio_to_video

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


if __name__ == '__main__':
    orig_vid_dir = '/home/video.mp4'
    
    ## function to extract frames
    frame_dir = '/home/frames/'
    os.makedirs(frame_dir, exist_ok=True)
    extract_frames(orig_vid_dir, frame_dir)
    
    ## Object detection model inference
    # sudo CUDA_VISIBLE_DEVICES=0 python3 detect.py --source path_to_frames --conf 0.25 --weights runs/train/train_cliff_sample2/weights/best.pt --save-txt
    # output is saved in a folder located  at ./runs/detect/
    #### blur script 
    # python3 4_blur_faces.py -i path_to_frames -l path_to_labels_txt -o path_to_saved_output -r 0.9
    ## missing files script

    # Command 1: Running detect.py with sudo and setting CUDA_VISIBLE_DEVICES
    detect_command = [
        'sudo',
        'CUDA_VISIBLE_DEVICES=0',
        'python3', 
        'detect.py', 
        '--source', 'path_to_frames', 
        '--conf', '0.25', 
        '--weights', 'runs/train/train_cliff_sample2/weights/best.pt', 
        '--save-txt'
    ]

    # Using subprocess to run the command
    subprocess.run(' '.join(detect_command), shell=True)

    # Command 2: Running 4_blur_faces.py
    blur_command = [
        'python3', 
        '4_blur_faces.py', 
        '-i', 'path_to_frames', 
        '-l', 'path_to_labels_txt', 
        '-o', 'path_to_saved_output', 
        '-r', '0.9'
    ]

    # Using subprocess to run the command
    subprocess.run(blur_command)


    ## extract audio from original video
    audio_dir = '/home/audio/'
    os.makedirs(audio_dir, exist_ok=True)
    extract_audio(orig_vid_dir, audio_dir)
    
    ## get text from audio
    text_dir = '/home/text/'
    os.makedirs(text_dir, exist_ok=True)
    asr_model = audio_transcriber("large", True) # init ASR model
    # run model
    text, words_timestamp = asr_model(os.path.join(audio_dir, "audio.wav"))
    # save transcript
    with open(os.path.join(text_dir, 'transcript.txt'),"w+") as f:
        f.write(text)
    # save python list containing indiv sent and time stamp
    with open(os.path.join(text_dir, 'sent_timestamp.json'),"w+") as f:
        json.dump(words_timestamp, f)
    
    # get pii from text
    text_pii_dir = '/home/text_pii/'
    os.makedirs(text_pii_dir, exist_ok=True)
    alw_tag = ["PERSON", "ORG", "LOC"]
    pii_model = NER_model()
    extract_pii(text_dir, text_pii_dir, pii_model, alw_tag)
    
    ## beep audio
    filename = 'text_pii'
    beep_aud_dir = "/home/beep_aud/"
    os.makedirs(beep_aud_dir, exist_ok=True)
    print("Replacing by beep sound..")
    with open(text_pii_dir+filename+"_redact.json") as f:
        mute_segments_list = json.load(f)
    mute_segments_list.sort(key=lambda x:x[2])
    mute_segments(audio_dir+"audio.wav",mute_segments_list, beep_aud_dir+"beep.wav")

    ## create video from result frames
    input_res_frames = "/home/...."
    output_video = "/home/output_video.mp4"
    create_video_from_frames(input_res_frames, output_video)
    
    ## merge audio and video 
    output_aud_vid = "/home/video_with_audio.mp4"
    join_audio_to_video(output_video, beep_aud_dir+"beep.wav", output_aud_vid)
    
