from tools.utils.youtube import youtube as yt_util
from tools.utils.openpose import openpose as op_util
from tools.utils.video import flip_movies
from tools.utils.video import zoom_movies
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Program in which a user can specify the path to a .csv file, download them, and 'clean them up' for use with st-gcn")

    parser.add_argument('-d', '--data_path', help='Path to data', default = "/data")
    parser.add_argument('-a', '--augment', help='To augment the inputed videos, or not', default = 'True')
    args = parser.parse_args()

    data_path = args.data_path
    augment = args.augment


    list_train_val = ['train', 'val']
    for alt in list_train_val:
        csv_path = os.path.join(data_path, os.path.join('youtube', 'kinetics_{}.csv'.format(alt)))
        yt = yt_util(csv_path = csv_path, data_path = data_path)
        videos_cleaned, videos_augment, videos_keypoints = yt.get_youtube_videos()
    
    # For when testing the program, and do not want to download the videos 
    #videos_cleaned, videos_augment, videos_keypoints = "{}/youtube/videos_clean".format(data_path), "{}/youtube/videos_clean_augmented".format(data_path), "{}/youtube/videos_clean_keypoints".format(data_path) 
    
    # Currently only flipping the videos, tho probably the most relevant augmentation. 
        if augment:
            flip_movies(videos_path_input = videos_cleaned, videos_path_output = videos_augment)
            zoom_movies(videos_path_input = videos_cleaned)


    #Testing on test videos in another folder
    #zoom_movies(videos_path_input = "/home/stian/zooming_test")
    

        op = op_util(data_path = data_path, data_videos_clean = videos_cleaned, data_videos_keypoints = videos_keypoints)
        op.openpose()
        op.openpose_skeleton_to_stgcn(train_or_val = alt)

