import skvideo.io
import numpy as np
import cv2
import os
import sys
import torch
from torchvision import transforms, utils
import torchvision.transforms as transforms
import subprocess
from PIL import Image

import time

def video_info_parsing(video_info, num_person_in=5, num_person_out=2):
    data_numpy = np.zeros((3, len(video_info['data']), 18, num_person_in))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                    0))
    data_numpy = data_numpy[:, :, :, :num_person_out]

    label = video_info['label_index']
    return data_numpy, label

def get_video_frames(video_path):
    vread = skvideo.io.vread(video_path)
    video = []
    for frame in vread:
        video.append(frame)
    return video

def video_play(video_path, fps=30):
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1000/fps) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# videos_path: /home/stian/Master_thesis/st-gcn/data/youtube/videos_clean
def flip_movies(videos_path_input, videos_path_output, extension = ".mjpeg"):
    '''
        Data augmentation: Flip movies
    
        videos_path_input: where to find the videos to flip
                e.g. data_augmentation_path_input="st-gcn/data/youtube/videos_clean"

        videos_path_output: where to store the flipped videos
                e.g. data_augmentation_path_output="st-gcn/data/youtube/videos_clean_augmented

    '''
    # Output 
    frames_output = os.path.join(videos_path_output, "flipped_images")
    
    for video_name in os.listdir(videos_path_input):
        print("Flipping frames: {}".format(video_name), end='\r')
        # Input
        video_name_no_ext = video_name.split(".")[0]
        video_ext = video_name.split(".")[1]
        video_f_path_input = os.path.join(videos_path_input, video_name)

        # Convert to .mjpeg, necessary because cv2 is not able to read mp4 files, weird tho because on inspecting the files they have the same CODECs?
        new_video_full_path = os.path.join(videos_path_input, video_name_no_ext) + extension
        os.rename(video_f_path_input, new_video_full_path)

        frames_f_output = os.path.join(frames_output, video_name_no_ext)

        # Want to store the flipped videos with the other videos (that is being flipped), such that it is easy when running them through openpose
        # Necessary with "__" because of string splitting later in the program, in which we need to differentiate between the label, and the fact the file
        # is 'flipped' 
        video_flipped_name = "{}__{}".format(video_name_no_ext, "flipped")
        video_zoomed_name = "{}__{}".format(video_name_no_ext, "zoomed")

        video_flipped_f_name = "{}.{}".format(video_flipped_name, "mp4")
        video_zoomed_f_name = "{}.{}".format(video_zoomed_name, "mp4")

        video_flipped_f_path = os.path.join(videos_path_input, video_flipped_f_name)
        video_zoomed_f_path = os.path.join(videos_path_input, video_zoomed_f_name)

        # Flip the movies' frames, and store them in a "Flipped" folder
        flip_frames(video_name_no_ext = video_name_no_ext, 
                    video_f_path_input = new_video_full_path, 
                    frames_f_path_output = frames_f_output)

        frames_to_video(input_path = frames_f_output, output_path = video_flipped_f_path, output_name = video_name_no_ext, video_ext = video_ext)

def flip_frames(video_name_no_ext, video_f_path_input, frames_f_path_output):
    '''
        video_name: e.g. "0EDCBPPstwY-jumping_jacks"

        video_f_path_input: e.g. "st-gcn/data/youtube/videos_clean/2d1rnOm9IGQ-cup_song.mjpeg"
        video_f_path_output: e.g. "st-gcn/data/youtube/videos_clean/2d1rnOm9IGQ-cup_song__flipped.mjpeg"
        frames_f_path_output: e.g. "st-gcn/data/youtube/videos_clean_augmented/flipped_images/2d1rnOm9IGQ-cup_song"
    '''
    if not os.path.isfile(video_f_path_input):
        print("This file does not exists {}, please try again \n".format(video_f_path_input))
        return

    if not os.path.isdir(frames_f_path_output):
        os.makedirs(frames_f_path_output)


    vid = cv2.VideoCapture(video_f_path_input)
    success, image = vid.read()

    if not success:
        print("------------------ Failed reading video -------------------------\n")
    count = 0
    
    while success:
        # Writes the "original" frames to file, which is seemingly unnecessary
        # cv2.imwrite(os.path.join(video_f_path_output, ('frame%d.jpg' % count)), image)
    
        image_flipped = cv2.flip(image, 1)
        # print("image_flipped: ", image_flipped)
        cv2.imwrite(os.path.join(frames_f_path_output, ('frame%d_flipped.jpg' % count)), image_flipped)
    
        success, image = vid.read()
        count += 1    

def frames_to_video(input_path, output_path, output_name, video_ext):
    # output_name += "{}.{}".format("_flipped", video_ext) 
    # print("output_full: ", output_path)
    time.sleep(5)
    try:
        cmd = ("ffmpeg -framerate 30 -i " + input_path + "/frame%000d_flipped.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + output_path + " -loglevel quiet")
        p = subprocess.Popen(cmd, shell=True)
        
        # Wait until process is finished
        p.wait()
    except:
        print("Problems with the cmd command!")

def convert_from_mp4_to_avi(input_path, output_path):
    
    video_ext = input_path.split(".")[1]
    video_no_ext = input_path.split(".")[1]
    if video_ext != "avi":  # If not .avi --> convert to .avi
        #cmd = ("ffmpeg -i filename.mp4 -vcodec copy -acodec copy filename.avi")
        cmd = ("ffmpeg -i " + input_path + " -vcodec copy -acodec copy " + output_path)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()
    os.remove(input_path)
    new_path = video_no_ext + ".avi"
    return new_path

def zoom_movies(videos_path_input, videos_path_output):
    
    for video_name in os.listdir(videos_path_input):
        print("Zooming on {}".format(video_name), end='\r')
        video_name_no_ext, video_ext = video_name.split(".")
        video_f_path_input = os.path.join(videos_path_input, video_name)
    
        video_zoomed_name = "{}__{}".format(video_name_no_ext, "zoomed")
        video_zoomed_f_name = "{}.{}".format(video_zoomed_name, "mp4")
        video_zoomed_f_path = os.path.join(videos_path_output, video_zoomed_f_name)
    
        try:
            cmd = ("ffmpeg -i " + video_f_path_input + " -vf 'scale=1.1*iw:-1, crop=iw/1.1:ih/1.1' " + video_zoomed_f_path + ' -loglevel quiet')
            p = subprocess.Popen(cmd, shell=True)
            p.wait()
        except:
            print("Zooming went wrong.")

def clean_video(video_fpath_inp, video_fpath_out, breadth = 340, height = 256, frate = 30):
    #cmd = ("ffmpeg -i " + video_fpath_inp + " -vf scale=340:256 -r 30 " + video_fpath_out)
    cmd = ('ffmpeg -i {} -s {}:{} -r {} {} -loglevel quiet'.format(video_fpath_inp, breadth, height, frate, video_fpath_out))
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait()



            
