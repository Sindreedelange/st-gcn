from tools.utils.youtube_util import youtube_util as yt_util
from tools.utils.openpose_util import openpose_util as op_util
from tools.utils.video import flip_movies
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Program in which a user can specify the path to a .csv file, download them, and 'clean them up' for use with st-gcn")

    parser.add_argument('-p', '--csv_path', help='Path to .csv file', default = "st-gcn/data/youtube/download_videos.csv")
    parser.add_argument('-d', '--data_path', help='Path to data', default = "st-gcn/data")
    parser.add_argument('-a', '--augment', help='To augment the inputed videos, or not', default = 'True')
    args = parser.parse_args()

    csv_path = args.csv_path
    data_path = args.data_path
    augment = args.augment

    yt = yt_util(csv_path = csv_path, data_path = data_path)
    videos_cleaned, videos_augment, videos_keypoints = yt.get_youtube_videos()
    
    # For when testing the program, and do not want to download the videos 
    # videos_cleaned, videos_augment, videos_keypoints = "{}/youtube/videos_clean".format(data_path), "{}/youtube/videos_clean_augmented".format(data_path), "{}/youtube/videos_clean_keypoints".format(data_path) 
        
    # Currently only flipping the videos, tho probably the most relevant augmentation. 
    if augment:
        flip_movies(videos_path_input = videos_cleaned, videos_path_output = videos_augment)

    op = op_util(data_path = data_path, data_videos_clean = videos_cleaned, data_videos_keypoints = videos_keypoints)
    op.openpose()
    op.openpose_skeleton_to_stgcn()

