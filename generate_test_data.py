from tools.utils.pose_estimator_test import pose_estimator_test
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

    verify_directory(videos_raw)
    verify_directory(videos_cleaned)
    verify_directory(videos_augment)
    verify_directory(videos_keypoints)    

    #for video in os.listdir(videos_raw):
    #    cur_vid_inp = os.path.join(videos_raw, video)
    #    cur_vid_outp = os.path.join(videos_cleaned, video)
    #    print("Cleaning {}".format(video), end='\r')
    #    clean_video(cur_vid_inp, cur_vid_outp)
    #print("\nCleaning done\n\n")
    #
    #if augment:
    #    flip_movies(videos_path_input = videos_cleaned, videos_path_output = videos_augment)
    #    zoom_movies(videos_path_input = videos_cleaned, videos_path_output = videos_augment)    

    pose_est = pose_estimator_test(data_path = data_path, 
                                data_videos_clean = videos_cleaned, 
                                data_videos_keypoints = videos_keypoints)
    pose_est.run()
    pose_est.skeleton_to_stgcn()