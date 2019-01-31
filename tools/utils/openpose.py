from pathlib import Path
import json

from natsort import natsorted
from pathlib import Path
import json

import os
import subprocess
import time
import psutil

from .file_util import *
# from views.output_messages import *

class openpose():

    def __init__ (self,  
                    data_path,
                    data_videos_clean,
                    data_videos_keypoints,
                    label_text_file = "resource/kinetics_skeleton/label_name_reduced.txt",
                    openpose_bin_path = "openpose/build/examples/openpose/openpose.bin", 
                    model_folder = "openpose/models/"):        
        '''
            data_path: String - Path to data

            data_videos_clean: String - path to (clean) videos to be ran through openpose
                                aka. extract keypoints 
            data_videos_keypoints: String - path to where each videos' keypoints are to be stored
            
            openpose_bin_path: Path to find the '.bin' file necessary to run the videos through openpose
            model_folder: Because we are not running our scripts from inside '/openpose' we need to define where one can find the models used for mapping the skeletons
            data_json_skeleton_train = String - Path to where skeleton files (.json) for the training set should be stored
            data_json_skeleton_validation = String - Path to where skeleton files (.json) for the validation set should be stored
            data_json_description_train_path = String - Path to the 'description'/'summary' file for each skeleton file in the training set
            data_json_description_validation_path = Path to the 'description'/'summary' file for each skeleton file in the validation set):
        '''
        self.data_kinetics_skeleton = "{}/Kinetics/kinetics-skeleton".format(data_path)

        self.data_videos_clean = data_videos_clean
        self.data_videos_keypoints = data_videos_keypoints

        self.label_text_file = label_text_file
        self.openpose_bin_path = os.path.join(".", openpose_bin_path)
        self.model_folder = model_folder

        self.data_json_skeleton_train = "{}/kinetics_train_reduced".format(self.data_kinetics_skeleton)
        self.data_json_skeleton_validation = "{}/kinetics_val_reduced".format(self.data_kinetics_skeleton)
        self.data_json_description_train_path = "{}/kinetics_label_reduced_train.json".format(self.data_kinetics_skeleton)
        self.data_json_description_validation_path = "{}/kinetics_label_reduced_val.json".format(self.data_kinetics_skeleton)

        # Verify that all of the default folders exists, if not, make them
        verify_directory(self.data_videos_keypoints)
        verify_directory(self.data_videos_clean)
        verify_directory(self.data_json_skeleton_train)
        verify_directory(self.data_json_skeleton_validation)

        # Load the .json 'description files
        self.data_json_description_train = file2dict(self.data_json_description_train_path)
        self.data_json_description_validation = file2dict(self.data_json_description_validation_path)

    
    def openpose(self):
        '''
            Runs through all of the (cleaned) downloaded videos from Youtube, check for duplicates (the existence of skeletonfiles with the same name),
            if they are not duplicates, store the skeletonfiles (gotten from Openpose), and rename that such that later work will be simpler.  
        '''
        clean_video_list = os.listdir(self.data_videos_clean)

        # Just for output to the user
        num_files = len(clean_video_list)
        counter = 0

        for file in clean_video_list:
            counter += 1

            print("\n --------------------------------------------------------------------- \n")
            print("Currently running {} through OpenPose \t {}/{}".format(file, counter, num_files))
            print("\n --------------------------------------------------------------------- \n")
            # Input video full path
            video_path_full = os.path.join(self.data_videos_clean, file)
       
            # The folder name, for each video's skeleton files, should not include the video extension
            filename_no_extension = file.split(".")[0]

            duplicate_file = check_duplicates(folder_path = self.data_videos_keypoints, file_name = filename_no_extension)

            if not duplicate_file:
                output_path_full = os.path.join(self.data_videos_keypoints, filename_no_extension)

                successfull = False
                while not successfull:
                    successfull = self.run_video_through_openpose(input_f_path = video_path_full, output_f_path = output_path_full)

                self.rename_keypoints_files(file_f_path = output_path_full)
            else:
                duplicate_files_error_message(self.data_videos_keypoints, filename_no_extension)

    def run_video_through_openpose(self, input_f_path, output_f_path): 
        '''
            Run the cleaned videos from Youtube through openpose to get their skeletonfiles

            input_f_path: String - Full path to the file that should be put through Openpose
            output_f_path: String - Full path to where the new (skeletonfile) should be stored

            Return: Boolean - Whether or not the video was ran through openpose, successfully, or if it froze, such that the program had to be terminated
        '''
        cmd = (self.openpose_bin_path + " --video " + input_f_path + " --model_folder " + self.model_folder + " --write_json " + output_f_path + " --model_pose COCO --keypoint_scale 3")        
        parent = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait until process is finished - not relevant anymore because the program needs to keep running in order to stop Openpose, if it freezes
        # p.wait()

        # Problems with openpose freezing after x number of videos, so to combat this: 
            # Stop the process after y seconds, and try again 
        timeout_limit = 45
        successfull = True
        for _ in range(timeout_limit):
            time.sleep(1)
            if parent.poll() is not None:  # process just ended
                break
        else:
            # the for loop ended without break: timeout
            try:
                successfull = False
                parent_process = psutil.Process(parent.pid)
                for child in parent_process.children(recursive=True):  # or parent.children() for recursive=False
                    child.kill()
                parent_process.kill()
            except Exception:
                print("--------------------------------------------------------------------- \n")
                print("Failed trying to kill the process")
                print("--------------------------------------------------------------------- \n")

        return successfull
    
    @staticmethod
    def rename_keypoints_files(file_f_path):
        '''
            Rename skeleton files

            Going from:
                    'youtube extesion-label_00000000000x_keypoints.json' 
                    to
                    'x_keypoints.json' 
            Seeing as the folder contains information about both extension and label it is not necessary to have here

            file_f_path: String - full path to where the folder, containing all the skeleton files, is located 
        '''
        skeleton_files_list_sorted = natsorted(os.listdir(file_f_path))
        
        counter = 0

        for skeleton_file in skeleton_files_list_sorted:
            new_filename = str(counter) + '_keypoints.json'
        
            old_file = os.path.join(file_f_path, skeleton_file)
            new_file = os.path.join(file_f_path, new_filename)
            os.rename(old_file, new_file)

            counter += 1


    @staticmethod
    def get_num_labels(dir_path):
        '''
            Input path to directory which contains folders, each corresponding to a youtube video.
            Finding each video's label in their name, and counting them

            dir_path: String - path to directory containing the folder corresponding to a youtube video

            Return: Dictionary
                        label: label name
                            total: total occurences of that specific label
                            counter: used to compare with total to get the ratio
        '''
        file_list = os.listdir(dir_path)

        label_dict = {}
        for file in file_list:
            # Example: 7oqsc6ahHVI--cup_song__0__flipped --> Get 'cup_song'
            label = file.split("---")[1].split("__")[0]

            if label in label_dict:
                label_dict[label]['total'] += 1
            else:
                # New label, so initialize it with 'starting values' 
                label_dict[label] = {'total': 1, 'current': 0}
        
        return label_dict

    @staticmethod
    def train_or_val(label_counter_dict, label, train_val_ratio):
        '''
            Based on the inputed dictionary, return whether the data should be stored in train or validation data set

            label_counter_dict: Dictionary 
                label:
                    total: 
                    counter:
            label: String - name of the current label
            train_val_ratio: Double - the ratio to which compare the counter/total number to, which ultimately decides if train or validation

            Return: Boolean
                True: Training set
                False: Validation set

        '''
        # Divide the data between train and validation data set
        current = label_counter_dict[label]['current']
        total = label_counter_dict[label]['total']

        ratio = current/total

        return ratio < train_val_ratio 

    def openpose_skeleton_to_stgcn(self, train_val_ratio = 0.8, frame_limit = 300):
        '''
            "Translate" openpose skeletonfiles to one single skeletonfile which st-gcn accepts as input, for either training or validating

            output_path: String - Where to output both the translated skeleton files, and the corresponding label dictionaries
            train_val_ratio: Double - The ratio between training and validation data 
            frame_limit: Int - The limit on number of frames pr. video (the paper used 300, so we will be using the same)
                Relevant when the videos are < 10 seconds long
        '''    
        # Make sure that the downloaded files are separated between 'train' and 'val'
        test_val_ratio_dict = self.get_num_labels(self.data_videos_keypoints)

        for folder in os.listdir(self.data_videos_keypoints):
            #print("--------------------------------------------------------------------- \n")
            #print("Currently working on {}".format(folder))
            # print("--------------------------------------------------------------------- \n")
            
            # Make sure no more than 300 frames pr video 
            frame_counter = 0
        
            folder_name = os.path.join(self.data_videos_keypoints, folder)
        
            stgcn_data_array = []
            stgcn_data = {}

            # Get label name from folder name
            label = folder.split("---")[1].split("__")[0]
            # Corresponding Label Index from the label text file
            label_index = self.get_label_index(label)

            # True if the data should be part of the training set, False if it should be part of the validation set
            train = self.train_or_val(test_val_ratio_dict, label, train_val_ratio)

            if train:
                old_dictionary = self.data_json_description_train
                old_dictionary_path = self.data_json_description_train_path
                current_train_val_folder = self.data_json_skeleton_train
            else:
                old_dictionary = self.data_json_description_validation
                old_dictionary_path = self.data_json_description_validation_path
                current_train_val_folder = self.data_json_skeleton_validation
                 
            # Increase counter
            test_val_ratio_dict[label]['current'] += 1

            filename = folder + ".json"
            dest_path = os.path.join(current_train_val_folder, filename) # Store skeleton files here
        
            file_list_sorted = natsorted(os.listdir(folder_name))
            for file in file_list_sorted:
                frame_counter += 1

                filename_full_path = os.path.join(folder_name, file)
                # Get frame id from the filename --> Ultimately combining them all to one .json file
                frame_id = int(((filename_full_path.split('/')[-1]).split('.')[0]).split('_')[0]) 
            
                frame_data = {'frame_index': frame_id}
            
                # Load the .json files, one by one
                data = json.load(open(filename_full_path))
                skeletons = []
                for person in data['people']:
                    score, coordinates = [], []
                    skeleton = {}
                    keypoints = person['pose_keypoints_2d']
                    for i in range(0,len(keypoints),3):
                        coordinates +=  [keypoints[i], keypoints[i + 1]]
                        score += [keypoints[i + 2]]
                    skeleton['pose'] = coordinates
                    skeleton['score'] = score
                    skeletons += [skeleton]
                frame_data['skeleton'] = skeletons
                stgcn_data_array += [frame_data]  
            
                if frame_counter > frame_limit: # Do not exceed 300 frames
                    message = ("Too many frames in file: {} - limiting it to {}".format(filename, frame_limit))
                    print(message)
                    break
            
            # If < 300 frames - pad the dictionary by getting the x first frames in dictionary, where x = 300 - number of frames, and add them to the end
            if frame_counter < frame_limit:
                self.pad_video_dict(stgcn_data_array, frame_limit)
        
            # Append to old label dictionary
            old_dictionary[str(folder)] = {"has_skeleton": True, "label": label, "label_index": label_index} 
        
            stgcn_data['data'] = stgcn_data_array
            stgcn_data['label'] = label 
            stgcn_data['label_index'] = label_index 

            # Make sure that the label text file is updated
            update_label_list(label)
        
            #dict2file(dest_path, stgcn_data)
            # Store the skeleton file
            with open(dest_path, 'w') as outfile:
                json.dump(stgcn_data, outfile, indent=4)

            # dict2file(label_dest_path_filename, old_dictionary)
            # Store the .json dictionary: 'kinetics_label_reduced_val/train.json'
            with open(old_dictionary_path, 'w') as label_file:
                json.dump(old_dictionary, label_file, indent=4)

    def pad_video_dict(self, list_to_pad, frame_limit):
        '''
           Pad videos that are < frame_limit frames, so that to fully utilize the model by optimizing usage of the dataset

           Calculate number of frames to pad with = ratio, then append x first frames to the end of the list, where x = ratio

           list_to_pad: List containing keypoint information for each frame in the video 
        '''
        # Get the number of frames in the list
        num_frames = list_to_pad[-1]['frame_index']
        # Find the ratio between number of frames and 300
        ratio = frame_limit - num_frames
        # Get the x first frames in dictionary, where x = ratio, and add them to the end
        counter = 0
        while counter < (ratio - 1):
            num_frames += 1
            # Need to make a copy in order to not make changes to the original frame
            data_to_append = list_to_pad[counter].copy()
            # Edit the frame index, such that it is not the frame index of the copied frame
            data_to_append['frame_index'] = num_frames
            # Append it to the end
            list_to_pad.append(data_to_append)
            counter += 1


    def get_label_index(self, class_name):
        '''
            Get the a specific label's label index from a text file, specifying all of the different classes

            class_name: String - name of the class label one wishes to get the corresponding label_index
        '''
        class_name_file = open(self.label_text_file)

        counter = 0
        label_index = 0
        for line in class_name_file:
            if compare_strings(class_name, line):
                label_index = counter
                break
            counter += 1
        return label_index


    def generate_numpy_and_pkl(self, path_to_skeletons="st-gcn/data/Kinetics/kinetics-skeleton/", path_gendata="st-gcn/tools/kinetics_gendata.py"):
        '''
            ***********
            Deprecated - Use 'python st-gcn/tools/kinetics_gendata.py --data_path st-gcn/data/Kinetics/kinetics-skeleton/'
            ***********

            Generate new numpy and pickle files for the training part

            path_to_skeleton: path to where the different folders are (train, val)
            path_gendata: path to where st-gcn's gendata method is located, which generates the aformentioned numpy and pickle files
        '''

        cmd = ("python " + path_gendata + " --data_path " + path_to_skeletons)
        p = subprocess.Popen(cmd, shell=True)

        # Wait until process is finished
        p.wait()

    @staticmethod
    def json_pack(snippets_dir, video_name, frame_width, frame_height, label='unknown', label_index=-1):
        sequence_info = []
        p = Path(snippets_dir)
        for path in p.glob(video_name+'*.json'):
            json_path = str(path)
            frame_id = int(path.stem.split('_')[-2])
            frame_data = {'frame_index': frame_id}
            data = json.load(open(json_path))
            skeletons = []
            for person in data['people']:
                score, coordinates = [], []
                skeleton = {}
                keypoints = person['pose_keypoints_2d']
                for i in range(0, len(keypoints), 3):
                    coordinates += [keypoints[i]/frame_width, keypoints[i + 1]/frame_height]
                    score += [keypoints[i + 2]]
                skeleton['pose'] = coordinates
                skeleton['score'] = score
                skeletons += [skeleton]
            frame_data['skeleton'] = skeletons
            sequence_info += [frame_data]
    
        video_info = dict()
        video_info['data'] = sequence_info
        video_info['label'] = label
        video_info['label_index'] = label_index
    
        return video_info