
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

import subprocess
from utils.file_util import verify_directory, update_num_classes_yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_kinetics import Feeder_kinetics
sys.path.append("/home/stian/Master_thesis_fork/st-gcn/tools")

toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        num_person_in=5,  #observe the first 5 persons 
        num_person_out=1,  #then choose 2 persons with the highest score 
        max_frame=300):

    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, 18, num_person_out))

    for i, s in enumerate(sample_name):
        data, label = feeder[i]
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='data/Kinetics/Kinetics/kinetics-skeleton/')
    parser.add_argument(
        '--out_folder', default='data/Kinetics/Kinetics/kinetics-skeleton/')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        # Path to skeleton files folder
        data_path = '{}/kinetics_{}_reduced'.format(arg.data_path, p)
        # Path to json files
        label_path = '{}/kinetics_label_reduced_{}.json'.format(arg.data_path, p)
        # Files to generate
        data_out_path = '{}/{}_data.npy'.format(arg.out_folder, p)
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

        verify_directory(arg.out_folder)

        gendata(data_path, label_path, data_out_path, label_out_path)
        print("\nUpdating number of classes \n")
        update_num_classes_yaml()