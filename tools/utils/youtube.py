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
import pickle


class youtube():
    '''
        Util class to be used for downloading and 'cleaning' Youtube videos, and updating necessary files there after (num classes, class names) 
    '''

    def __init__ (self,
                csv_path,
                data_path,
                data_augmentation=True):
        '''
            csv_path: String - path to csv file used to download Youtube videos (max 10 seconds)
                url: String - Full URL to Youtube video
                start: Datetime (hh:mm:ss) - Where to start the recording when cutting the video
                stop: Datetime (hh:mm:ss) - Where to end the recording when cutting the video
                label: String - Class label to assign the corresponding video clip

            data_folder: String - Path to data
            data_augmentation: Boolean - To flip the videos, or not, that is the question (pretty simple, seeing as it doubles the size of the dataset)
        '''
        self.csv_path = csv_path
        self.data_augmentation = data_augmentation

        self.data_youtube_folder = "{}/original_data".format(data_path)

        self.data_videos_download = "{}/videos".format(self.data_youtube_folder)
        self.data_videos_clean = "{}/videos_clean".format(self.data_youtube_folder)
        self.data_videos_augmentation_path_output = "{}/videos_clean_augmented".format(self.data_youtube_folder)
        self.data_videos_keypoints = "{}/videos_clean_keypoints".format(self.data_youtube_folder)

        # Verify that all of the default folders exists, if not, make them
        # Reset all of the default folders
        reset_directory(self.data_videos_download)
        reset_directory(self.data_videos_clean)
        reset_directory(self.data_videos_augmentation_path_output)
        reset_directory(self.data_videos_keypoints)


    def get_youtube_videos(self):
        '''
            Download and clean youtube videos based on a csv file: [url, start, stop, label]

            Return: Tuple - (Path to the 'clean' videos, Path to where the augmented frames should be stored)
        '''

        # Read the inputed csv file
        try:
            videos = pd.read_csv(self.csv_path)
        except Exception as e:
            print("Invalid .csv file - please try again")
            print("Error message: {}".format(e))
            sys.exit(1)

        videos = videos.values

        num_videos = len(videos)
        counter = 0

        list_unsuccessfull_vids_downloaded = []

        # Refactor to separate method?
        for video in videos:
            counter += 1
            print("File {}/{}".format(counter, num_videos), end='\r')
            
            # Remember: [url, start, stop, label]
            label = video[0]
            label = "_".join(label.split(" "))

            extension = video[1]

            full_link = 'https://www.youtube.com/watch?v={}'.format(extension)

            # Name the files " 'youtube extension'-'label name'.avi (switched from mp4 because of codec problems) "
            try:
                video_name = extension + "---" + label + ".mp4"
            except Exception:
                print("Unable to extract video name (aka.splitting on '=') from url: {}\nMoving on to next file".format(extension))
                continue

            video_f_path = os.path.join(self.data_videos_download, video_name)

            # Check if necessary to download the video, or just run it through 'cleaning'
            duplicate_file = check_duplicates(folder_path = self.data_videos_download, file_name = video_name)
            if not duplicate_file:
                print("Downloading {}".format(video_f_path))
                # Download videos
                self.download_youtube_video(video_download_f_path = video_f_path, url = full_link)

                _ = update_label_list(label=label)
            else:
                duplicate_files_error_message(output_folder = self.data_videos_download, file_name = video_name)

            # Clean videos
            self.clean_youtube_video(video, video_path = video_f_path, output_folder = self.data_videos_clean)
            try:
                os.remove(video_f_path)
            except:
                print("Could not delete {} - moving on".format(video_f_path))
                list_unsuccessfull_vids_downloaded.append(full_link)
        
        with open('work_dir/unsuccessfull_vid_downloads.txt', "w") as f:
            for s in list_unsuccessfull_vids_downloaded:
                f.write(str(s) +"\n")
        
        return self.data_videos_clean, self.data_videos_augmentation_path_output, self.data_videos_keypoints
        
    @staticmethod
    def download_youtube_video(video_download_f_path, url):
        '''
            Download the Youtube video corresponding to the given url, and store it using the inputed path

            video_download_f_path: String - (Full) Path to where the video should be stored
            url: String - Youtube URL for the video
        print("Extension: {}".format(url))
        '''
        # '-f' determines the format of the video (= mp4), '-o' determines the output folder (and the files' name)
        cmd = ("youtube-dl " + url + " -f mp4 -o " + video_download_f_path)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait until process is finished
        p.wait()

    def clean_youtube_video(self, video, video_path, output_folder):
        '''
            Use FFMPEG to clean the downloaded Youtube videos: 
                Framerate = 30
                Resolution = 340x256

            video: csv information [label, url, start, stop]
            video_path: full path to locate the video to be cleaned (tbc)
            output_folder: Where to store the cleaned video
        '''            
        start, duration = self.get_video_duration(video)

        # Seeing as the cleaned file should be named the same as the uncleaned file
        video_name = video_path.split("/")[-1]

        # Where to store the file
        output_file_f_path = self.find_available_name(output_folder = output_folder, video_name = video_name)

        # ffmpeg to cut the video
        cmd = ("ffmpeg -i " + video_path + " -ss " + start + " -vf scale=340:256 -framerate 30 -t " + str(duration) + " " + output_file_f_path + " -loglevel quiet")
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #! ffmpeg -i $video_in_path -ss $start -vf scale=340:256 -framerate 30  -t $duration $save_path_ffmpeg_full

        p.wait()

    @staticmethod
    def find_available_name(output_folder, video_name):
        '''
            Finding available names for "cleaned" outtakes from clips

            Useful when extracting multiple clips from one specific youtube clip

            output_folder: Folder in which the file is to be stored
            video_name: Name of the file to be stored

            Return: Path - Full, valid, output path for the file to-be-cleaned
        '''
        duplicate = True
        counter = 0
    
        string_to_append = ("__{}".format(counter))

        video_name_no_ext, video_ext = video_name.split(".")
    
        video_name_no_ext += string_to_append
        video_name_no_ext += (".{}".format(video_ext))
        video_name = video_name_no_ext
    
        duplicate = check_duplicates(output_folder, video_name)
        # 4vP1Z5tmww4-cup_song__0.mp4
    
        while duplicate:
            old_counter = counter
            counter += 1
            video_name, video_name_replace_counter = video_name.split("__")
            video_name_replace_counter = video_name_replace_counter.replace(str(old_counter), str(counter), 1)
            video_name += ("__{}".format(video_name_replace_counter))
            duplicate = check_duplicates(output_folder, video_name)
        
        return os.path.join(output_folder, video_name)

    @staticmethod
    def get_video_duration(video):
        '''
            Get the duration of a video, based on format from csv file: [url, start, stop, label]
            Note: Currently not able to pass in hours, only minutes and seconds

            video: array with 4 elements: [url, start, stop, label]

            returns
                start: At which point the relevant part of the videos start, such that the cleaned video start here - hh:mm:ss  
                duration: How long does the relevant action last, such that the cleaned video can stop here - hh:mm:ss - [1, 11] seconds
        '''
        start_time = video[2]
        end_time = video[3]
        duration = end_time - start_time

        start_time = humanize_time(start_time)

        return (start_time, duration)