from tools.utils.pose_estimator import pose_estimator
from tools.utils.file_util import verify_directory
from tools.utils.video import flip_movies, zoom_movies, clean_video
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Program that makes a test set out of (already downloaded) video files")

    parser.add_argument('-d', '--data_path', help='Path to data', default = "data")
    parser.add_argument('-a', '--augment', help='To augment, or not', default = 'True')
    args = parser.parse_args()

    data_path = args.data_path
    augment = args.augment

    data_path_test = os.path.join(data_path, 'test')

    videos_raw = os.path.join(data_path_test, 'raw')
    videos_cleaned = os.path.join(data_path_test, 'cleaned')
    videos_augment = os.path.join(data_path_test, 'augmented')
    videos_keypoints = os.path.join(data_path_test, 'keypoints')
    videos_skeletons = os.path.join(data_path_test, 'kinetics-skeleton')

    verify_directory(videos_raw)
    verify_directory(videos_cleaned)
    verify_directory(videos_augment)
    verify_directory(videos_keypoints)
    verify_directory(videos_skeletons)
    

    for video in os.listdir(videos_raw):
        cur_vid_inp = os.path.join(videos_raw, video)
        cur_vid_outp = os.path.join(videos_cleaned, video)
        print("Cleaning {}".format(cur_vid_inp), end='\r')
        clean_video(cur_vid_inp, cur_vid_outp)
    
    if augment:
        flip_movies(videos_path_input = videos_cleaned, videos_path_output = videos_augment)
        zoom_movies(videos_path_input = videos_cleaned)    

    pose_est = pose_estimator(data_path = data_path, 
                                videos_clean = videos_cleaned, 
                                videos_keypoints = videos_keypoints, 
                                videos_skeletons = videos_skeletons, 
                                train_val_test = 'test')
    pose_est.pose_estimating()
    pose_est.skeleton_to_stgcn()