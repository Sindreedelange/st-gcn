#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
import pandas as pd 
import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

from tools.views.output_messages import *

import json
import copy

from scipy.special import softmax
from tools.utils.file_util import get_label_text_file, compare_strings, verify_directory


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        message = "Loading model"
        print_generic_message(message)

        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label, sample_name in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluator = None, evaluation=True):
        message = "Testing model"
        print_generic_message(message)

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        summary_path = os.path.join(self.arg.work_dir, 'summary')
        verify_directory(summary_path)

        train_inference_tot_fname = "train_inference.csv"
        train_inference_fpath = os.path.join(summary_path, train_inference_tot_fname)

        for data, label, sample_name in loader:
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # inference
            with torch.no_grad():
                output = self.model(data)

            result_frag.append(output.data.cpu().numpy())
            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

            self.evaluate.inference_full_add_row(file_name = sample_name, label = label, predicted_vals = output)

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

        # Process inference information stored during testing and summarize for each class
        message = "Summarizing inference information"
        print_generic_message(message)
        self.evaluate.summarize_inference_full()

    def save_sum_summarised_csv(self, path, sum_dict):
        gen_sum_dict = ['Correct', 'Incorrect', 'Sum']
        df = pd.DataFrame(columns=gen_sum_dict)

        for k, v in sum_dict.items():
            df.loc[k] = v
        df.to_csv(path)

    def get_summarized_dict(self, path):
        summary = pd.read_csv(path)

        # Get the class names from the already summarized file
        class_name_list = list(summary['Actual Label'].unique())
        sum_dict = dict.fromkeys(class_name_list, {})
        gen_sum_dict = {'Correct': 0, 'Incorrect': 0, 'Sum': 0}
        for k, v in sum_dict.items():
            sum_dict[k] = copy.deepcopy(gen_sum_dict)
        
        for _, row in summary.iterrows():
            y_pred = row['Predicted Label']
            y_true = row['Actual Label']
            file_name = row['File name']            
            if compare_strings(y_pred, y_true):
                sum_dict[y_true]['Correct'] += 1
            else:
                sum_dict[y_true]['Incorrect'] += 1
            sum_dict[y_true]['Sum'] += 1  

        return sum_dict

    def save_to_csv(self, path, file_names, labels, predicted_values):
        df = pd.DataFrame()
        if os.path.isfile(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=['File name', 'Actual Label' , 'Predicted Label', 'Predicted Values %'])
        for i in range(len(file_names)):
            predicted_values_list = [v.item() for v in predicted_values[i]]
            preds_perc = self.get_predictions_in_percentage(predicted_values_list)
            value, key = self.dict_max_value(dic = preds_perc)
            
            actual_label = self.get_label_name(labels[i].item())

            new_row = [file_names[i], actual_label, key, preds_perc]
            df.loc[len(df)] = new_row
        df.to_csv(path, index=False)

    
    def get_label_name(self, index):
        lfile = get_label_text_file()
        lines = lfile.readlines()
        try:
            val = int(index)
            return lines[val].rstrip()
        except Exception:
            print("Got a string instead of an index? \n{}".format(index))
            return index

    def dict_max_value(self, dic):
        return max(zip(dic.values(), dic.keys()))
    
    def get_predictions_in_percentage(self, li):
        min_value = np.amin(li)
        preds_pos = li + (min_value ** 2) # So it is a positive number
        preds_soft = softmax(preds_pos)
        top5 = preds_soft.argsort()[-5:][::-1]
        
        zipped = {}
        counter = 0
        for el in top5:
            counter += 1
            label_name = self.get_label_name(index = el)
            zipped[label_name] = round((preds_soft[el]*100), 3)

        return zipped

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
