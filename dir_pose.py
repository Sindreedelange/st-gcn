from tools.utils.pose_estimator import pose_estimator
import os
import argparse
import time
import subprocess
import json

from natsort import natsorted


from utils.file_util import file2dict, compare_strings, check_duplicates

def get_label_index(class_name, label_text_file = 'resource/kinetics_skeleton/label_name.txt'):
    '''
        Get the a specific label's label index from a text file, specifying all of the different classes
        class_name: String - name of the class label one wishes to get the corresponding label_index
    '''
    class_name_file = open(label_text_file)
    counter = 0
    label_index = 0
    for line in class_name_file:
        if compare_strings(class_name, line):
            label_index = counter
            break
        counter += 1
    return label_index

def skeleton_to_stgcn(input_path, output_path, phase, frame_limit = 299):
    '''   
    "Translate" skeletonfiles to one single skeletonfile which st-gcn accepts as input, for either training or validating   
    
    output_path: String - Where to output both the translated skeleton files, and the corresponding label dictionaries   
    train_val_ratio: Double - The ratio between training and validation data    
    frame_limit: Int - The limit on number of frames pr. video (the paper used 300, so we will be using the same)   
        Relevant when the videos are < 10 seconds long
    
    '''
    unable_to_load_json_files_list = []
    folder_skeleton = os.path.join(output_path, 'kinetics_{}'.format(phase))
    if not os.path.isdir(folder_skeleton):
        os.makedirs(folder_skeleton)
    json_skeleton_fpath = os.path.join(output_path, 'kinetics_{}.json'.format(phase))
    json_skeleton_dictionary = file2dict(json_skeleton_fpath)

    counter = 0
    input_files = os.listdir(input_path)
    num_folders = len(input_files)
    for folder in input_files:
        counter += 1
        print("Interpreting keypoint files {}/{}".format(counter, num_folders))
        #print("--------------------------------------------------------------------- \n")
        #print("Currently working on {}".format(folder))
        # print("--------------------------------------------------------------------- \n")
        
        # Make sure no more than 300 frames pr video 
        frame_counter = 0
    
        input_fpath = os.path.join(input_path, folder)
    
        stgcn_data_array = []
        stgcn_data = {}
        # Get label name from folder name
        label = folder.split("---")[1].split("__")[0]
        # Corresponding Label Index from the label text file
        label_index = get_label_index(label)

        old_dictionary = json_skeleton_dictionary
        old_dictionary_path = json_skeleton_fpath
        current_folder = folder_skeleton
             
        # Increase counter
        filename = folder + ".json"
        output_folder_fpath = os.path.join(current_folder, filename) # Store skeleton files here
    
        file_list_sorted = natsorted(os.listdir(input_fpath))
        for file in file_list_sorted:
            frame_counter += 1
            filename_full_path = os.path.join(input_fpath, file)
            # Get frame id from the filename --> Ultimately combining them all to one .json file
            frame_id = int(((filename_full_path.split('/')[-1]).split('.')[0]).split('_')[0]) 
        
            frame_data = {'frame_index': frame_id}
        
            # Load the .json files, one by one
            try:
                data = json.load(open(filename_full_path))
            except Exception:
                unable_to_load_json_files_list.append(filename_full_path)
                continue

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
            pad_video_dict(stgcn_data_array, frame_limit)
    
        # Append to old label dictionary
        old_dictionary[str(folder)] = {"has_skeleton": True, "label": label, "label_index": label_index} 
    
        stgcn_data['data'] = stgcn_data_array
        stgcn_data['label'] = label 
        stgcn_data['label_index'] = label_index 
        # Make sure that the label text file is updated
    
        #dict2file(dest_path, stgcn_data)
        # Store the skeleton file
        try: 
            with open(output_folder_fpath, 'w') as outfile:
                json.dump(stgcn_data, outfile, indent=4)
        except Exception:
            print("Unable to open json file, or dump {} - moving on".format(outfile))

        # dict2file(label_dest_path_filename, old_dictionary)
        # Store the .json dictionary: 'kinetics_label_reduced_val/train.json'
        with open(old_dictionary_path, 'w') as label_file:
            json.dump(old_dictionary, label_file, indent=4)
    mode = 'a' if os.path.exists('unable_to_load_json_files.txt') else 'w'
    with open('unable_to_load_json_files.txt', mode) as f:
        for lines in unable_to_load_json_files_list:
            f.write(lines)
    f.close() 
def pad_video_dict(list_to_pad, frame_limit):
    '''
       Pad videos that are < frame_limit frames, so that to fully utilize the model by optimizing usage of the dataset
       Calculate number of frames to pad with = ratio, then append x first frames to the end of the list, where x = ratio
       list_to_pad: List containing keypoint information for each frame in the video 
    '''
    # Get the number of frames in the list
    try:
        num_frames = list_to_pad[-1]['frame_index']
    except Exception:
        print("The current file seemingly does not have any coordiantes")
        return
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

def run_pose_estimation(input_path, output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in os.listdir(input_path):
        input_fpath = os.path.join(input_path, file)
        filename = file.split('.')[0]
        output_fpath = os.path.join(output_path, filename)
        if check_duplicates(output_path, filename):
            print('\n Duplicate file: {}'.format(filename))
            continue

        pose_estimation_successfull = False

        while not pose_estimation_successfull:
            pose_estimation_successfull = pose_estimation(input_fpath = input_fpath, output_fpath = output_fpath)
        print("Successfully pose estimated file: {}".format(output_fpath))
        rename_keypoints_file(input_fpath = output_fpath)
    
def pose_estimation(input_fpath, output_fpath):
    cmd = ('python tf-pose-estimation/run_video.py --video ' + input_fpath + ' --output_json ' + output_fpath)
    
    parent = subprocess.Popen(cmd, shell=True)

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

def rename_keypoints_file(input_fpath):
    '''
        Rename skeleton files
        Going from:
                'youtube extesion-label_00000000000x_keypoints.json' 
                to
                'x_keypoints.json' 
        Seeing as the folder contains information about both extension and label it is not necessary to have here
        input_fpath: String - full path to where the folder, containing all the skeleton files, is located 
    '''
    skeleton_files_list_sorted = natsorted(os.listdir(input_fpath))
    
    counter = 0
    for skeleton_file in skeleton_files_list_sorted:
        new_filename = str(counter) + '_keypoints.json'
    
        old_file = os.path.join(input_fpath, skeleton_file)
        new_file = os.path.join(input_fpath, new_filename)
        os.rename(old_file, new_file)
        counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Program in which a user can specify the path to a .csv file, download them, and 'clean them up' for use with st-gcn")

    parser.add_argument('-p', '--phase', help='Phase: train og validation')
    args = parser.parse_args()

    phase = args.phase

    data_path = 'data'
    youtube_path = os.path.join(data_path, 'youtube')
    kinetics_path = os.path.join(data_path, 'Kinetics/kinetics-skeleton/')

    videos_clean_path = os.path.join(youtube_path, 'videos_clean_{}'.format(phase))
    videos_keypoints_path = os.path.join(youtube_path, 'videos_clean_keypoints_{}'.format(phase))

    #run_pose_estimation(input_path = videos_clean_path, output_path = videos_keypoints_path)
    skeleton_to_stgcn(input_path = videos_keypoints_path, output_path = kinetics_path, phase = phase)

