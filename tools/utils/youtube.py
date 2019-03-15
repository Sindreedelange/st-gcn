'''
    Author(s): Stian & Sindre
'''

import datetime
import pandas as pd
import numpy as np
import os
import subprocess
from .file_util import *
from tools.views.output_messages import *


class youtube():
    '''
        Util class to be used for downloading and 'cleaning' Youtube videos, and updating necessary files there after (num classes, class names) 
    '''

    def __init__ (self,
                data_path,
                data_augmentation=True,):
        '''
            csv_path: String - path to csv file used to download Youtube videos (max 10 seconds)
                url: String - Full URL to Youtube video
                start: Datetime (hh:mm:ss) - Where to start the recording when cutting the video
                stop: Datetime (hh:mm:ss) - Where to end the recording when cutting the video
                label: String - Class label to assign the corresponding video clip

            data_folder: String - Path to data
            data_augmentation: Boolean - To flip the videos, or not, that is the question (pretty simple, seeing as it doubles the size of the dataset)
        '''
        self.data_augmentation = data_augmentation

        self.data_raw = "{}/raw_videos".format(data_path)
        self.videos_cleaned = "{}/videos_cleaned".format(data_path)
        self.videos_augmented = "{}/augmentation".format(data_path)
        self.videos_keypoints = "{}/keypoints".format(data_path)

    def clean_videos(self):
        '''
            Clean videos, i.e. correct framerate and size

            Return: Tuple - (Path to the 'clean' videos, Path to where the augmented frames should be stored)
        '''

        # Read the inputed csv file
        videos = os.listdir(self.data_raw)
        num_videos = len(videos)
        counter = 0

        for video in videos:
            counter += 1
            print("File {}/{}".format(counter, num_videos), end='\r')

            video_fpath_input = os.path.join(self.data_raw, video)

            # Clean videos
            self.clean_video(video_path = video_fpath_input, output_folder = self.videos_cleaned)
        
        return self.videos_cleaned, self.videos_augmented, self.videos_keypoints

    def clean_video(self, video_path, output_folder):
        '''
            Use FFMPEG to clean the downloaded Youtube videos: 
                Framerate = 30
                Resolution = 340x256

            video_path: full path to locate the video to be cleaned (tbc)
            output_folder: Where to store the cleaned video
        '''            
        
        output_fpath = os.path.join(output_folder, video_path.split('/')[-1])
        # ffmpeg to cut the video
        cmd = ("ffmpeg -i " + video_path + " -vf scale=340:256 -framerate 30 " + output_fpath + " -loglevel quiet")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        p.wait()