from typing import Any
from flair.data import Sentence
from flair.nn import Classifier
import json
import glob
import os
import nltk

"""
Requirement

pip install flair nltk
"""

# # make a sentence
# sentence = Sentence('George Washington went to Washington.')

# # load the NER tagger
# tagger = Classifier.load('ner-ontonotes') # -large
 
# # run NER over sentence
# tagger.predict(sentence)

# # print the sentence with all annotations
# print(sentence)


def partial_match(string1, string2, threshold=0.5):
    distance = nltk.edit_distance(string1, string2)
    max_length = max(len(string1), len(string2))
    similarity = 1 - (distance / max_length)

    return similarity >= threshold


class NER_model:
    def __init__(self, tagger='ner-ontonotes-large'):
        '''
            Args:
                @text: string to classify for tags
            Result:
                output [text, (tag1, word1, start pos, end pos), (tag, word2,...),... ] ## skipping start and end pos for now
        '''
        self.model = Classifier.load(tagger)

    def __call__(self, text):
        sent = Sentence(text)
        # run NER over sentence
        self.model.predict(sent)
        # output list
        output = [text]
        # filling each entity as list 
        for entity in sent.get_spans('ner'):
            output.append({"tag":entity.tag,"word":entity.text})# ,entity.start_position, entity.end_position))

        return output 
        

def traverse_transcripts(base_dir, result_dir, pii_model, alw_tag):

    for file in glob.glob(os.path.join(base_dir,"*.json")):
        print("\n", "Tagging ...", file)
        filename = file.split('/')[-1].split('.')[0]

        # open json file for segments and timings
        with open(file) as f:
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
    # setup directories
    base_dir = "/home/ubuntu/prev_text" # audio_text
    result_dir = "/home/ubuntu/prev_text_pii" # text_pii
    os.makedirs(result_dir, exist_ok=True)

    # tag list
    alw_tag = ['PERSON','ORG','LOC']#, 'GPE', 'FAC', 'EVENT', 'LAW'

    # model
    pii_model = NER_model()

    # perform transcription
    traverse_transcripts(base_dir, result_dir, pii_model, alw_tag)
 