#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
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

from .io import IO

from tools.views.output_messages import print_generic_message
from tools.utils.evaluate import Evaluate

class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()


    def init_environment(self):
        
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.frozen = self.arg.freeze
        self.lr = self.arg.base_lr

    def load_evaluator(self):
        self.evaluate = Evaluate(work_dir = self.arg.work_dir)

    def load_optimizer(self):
        self.optimizer = self.arg.optimizer

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)
    
    # Note: Not the test method used when 'self.test()' is called - it is the one in 'training.py'
    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    # Note: Not the test method used when 'self.test()' is called - it is the one in 'training.py'
    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def store_eval_results(self, epoch, train_loss, val_loss, val_accuracy, inference_rows):
        cur_eval_folder = self.evaluate.get_eval_folder(epoch)
        inference_full = self.evaluate.get_inference_full_file()

        message = "Summarizing inference information"
        print_generic_message(message)

        for rows in inference_rows:
            for row in rows:
                inference_full.loc[len(inference_full)] = row

        # Currently unnecessary, because it basically gives ut the same information as the Confusion Matrix
        self.evaluate.summarize_inference_full(folder = cur_eval_folder, inference_frame = inference_full)

        self.evaluate.make_confusion_matrix(epoch = epoch, folder = cur_eval_folder, inference_frame = inference_full)
        self.io.print_log("Confusion Matrix generated")

        self.evaluate.store_loss_acc(train_loss = train_loss, val_loss = val_loss, accuracy = val_accuracy, folder = cur_eval_folder)
        self.io.print_log("Score summary .csv generated")

    def start(self):
        self.load_evaluator()
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        self.train_result_loss = list()

        # training phase
        if self.arg.phase == 'train':
            message = "Training started"
            print_generic_message(message)

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch
    
                # Unfreeze layers (and decrease the learning rate)
                if self.frozen and ((epoch + 1) >= (self.arg.num_epoch/10)):
                    self.io.print_log("Unfreezing")
                    self.frozen = False
                    self.unfreeze_all()
                
                # training
                self.io.print_log('Training epoch: {}/{}'.format(epoch, self.arg.num_epoch))
                loss_epoch_mean = self.train()
                self.train_result_loss.append(loss_epoch_mean)
                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch + 1))

                    val_accuracy, val_loss, inference_rows = self.test(epoch =  epoch + 1, evaluator = self.evaluate)

                    # Calculating mean loss over the eval interval
                    eval_interval_mean_loss = np.mean(self.train_result_loss)
                    # Clearing the list for calculating the mean over the next interval
                    self.train_result_loss.clear()

                    self.store_eval_results(epoch = epoch + 1, 
                                            train_loss = eval_interval_mean_loss, 
                                            val_loss = val_loss, 
                                            val_accuracy = val_accuracy,
                                            inference_rows = inference_rows)

                    self.io.print_log('Done.')
 
                        
            # self.io.print_log("Epoch {}/{}".format(self.meta_info['epoch'] + 1, self.arg.num_epoch))
            self.io.print_log("Training done - model saved at {}".format(os.path.join(self.arg.work_dir.split("/")[1], filename)))
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default='config/train.yaml', help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if true, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default='models/kinetics-st_gcn.pt', help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--freeze', type=str2bool, default=True, help='Freeze every layer but the last ones')
        #endregion yapf: enable

        return parser
