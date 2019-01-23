#!/usr/bin/env python
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io

from .io import IO
import tools
import tools.utils as utils

import matplotlib.pyplot as plt
import subprocess

from tools.utils.file_util import verify_directory
from tools.utils.openpose import openpose
from tools.utils import video as video_util


class Predict(IO):
    """
        Demo for Skeleton-based Action Recgnition
    """
    def start(self):

        openpose_bin_path = '{}/examples/openpose/openpose.bin'.format(self.arg.openpose)
        video_name = self.arg.video.split('/')[-1].split('.')[0]
        output_snippets_dir = 'data/openpose_estimation/snippets/{}'.format(video_name)
        output_sequence_dir = 'data/openpose_estimation/data'
        output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
        output_result_dir = self.arg.output_dir
        output_result_path = '{}/{}.avi'.format(output_result_dir, video_name)
        label_name_path = 'resource/kinetics_skeleton/label_name_reduced.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        print("\nPredicting on: {} \nUsing model: {}".format(video_name, self.arg.weights))
        
        # pose estimation - Running openpose on inputed file
        openpose_args = dict(
            video=self.arg.video,
            write_json=output_snippets_dir,
            display=0,
            render_pose=0, 
            model_pose='COCO')
        cmd = openpose_bin_path + ' '
        cmd += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
        # Delete potential old prediction folders
        shutil.rmtree(output_snippets_dir, ignore_errors=True)
        # Make output folder (basically the one that was just deleted)
        verify_directory(output_snippets_dir)
        p = subprocess.Popen(cmd, shell=True)
        # os.system(command_line)

        p.wait()
        # pack openpose ouputs - Get the video frames from the 'openposed video', which are to ran through the network (predicted on)
        video = video_util.get_video_frames(self.arg.video)

        height, width, _ = video[0].shape
        video_info = openpose.json_pack(
            output_snippets_dir, video_name, width, height)
        verify_directory(output_sequence_dir)
        with open(output_sequence_path, 'w') as outfile:
            json.dump(video_info, outfile)
        if len(video_info['data']) == 0:
            print('Can not find pose estimation results.')
            return
        else:
            print('Pose estimation complete.')

        # parse skeleton data
        pose, _ = video_util.video_info_parsing(video_info)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()

        # extract feature
        print('\nNetwork forward...')
        self.model.eval()
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)

        # Get prediction result
        predicted_label = label_name[label]
        
        print('Prediction result: {}'.format(predicted_label))
        print('Done.')
        
        predictions = output.sum(dim=3).sum(dim=2).sum(dim=1)
        predictions_np = predictions.data.cpu().numpy()

        # adding the minimum value in the array to every element 
        # to make everything positive
        min_value = np.amin(predictions_np)
        preidictions_np_positive = predictions_np - min_value

        # normalizing
        sum_elements = np.sum(preidictions_np_positive)
        predictions_norm = preidictions_np_positive / sum_elements


        labels = []
        values = []

        # Get top 5 predictions
        predictions_np_top5 = predictions_norm.argsort()[-5:][::-1]
        for label in predictions_np_top5:
            labels.append(label_name[label])
            values.append(predictions_norm[label]*100)

        print("Labels: ", labels)
        print("Values: ", values)


        # Matplot - barchart
        plt.figure(num=None, figsize=(18, 9), dpi=200, facecolor='w', edgecolor='k')
        index = np.arange(len(values))
        plt.bar(index, values)
        plt.xlabel('Class', fontsize=12, labelpad=10)
        plt.ylabel('Probability', fontsize=12, labelpad=10)
        plt.xticks(index, labels, fontsize=10, rotation=30)
        plt.title("Top 5 classes", fontsize=12, pad=10)
        # plt.show()
        chart_directory_name = 'charts'
        verify_directory(chart_directory_name)
        chart_name = 'probability.png'
        chart_f_path = os.path.join(chart_directory_name, chart_name)
        plt.savefig(chart_f_path, bbox_inches='tight')
        _ = subprocess.Popen(['gvfs-open', chart_f_path])


        # visualization
        print('\nVisualization...')
        label_sequence = output.sum(dim=2).argmax(dim=0)
        label_name_sequence = [[label_name[p] for p in l ]for l in label_sequence]
        edge = self.model.graph.edge
        images = utils.visualization.stgcn_visualize(
            pose, edge, intensity, video, predicted_label, label_name_sequence, self.arg.height)
        print('Done.')

        # save video
        print('\nSaving...')
        verify_directory(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path,
                                        outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The resulting video is stored in {}.'.format(output_result_path))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
            default='resource/media/jumping_jacks.mp4',
            help='Path to video')
        parser.add_argument('--openpose',
            default='openpose/build',
            help='Path to openpose')
        parser.add_argument('--output_dir',
            default='data/demo_result',
            help='Path to save results')
        parser.add_argument('--height',
            default=1080,
            type=int,
            help='Path to save results')
        parser.set_defaults(config='config/demo.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
