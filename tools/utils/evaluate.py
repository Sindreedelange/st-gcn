import os
import pandas as pd
import numpy as np
from scipy.special import softmax
import copy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from .file_util import dict_max_value, compare_strings, get_label_text_file

class Evaluate():

    def __init__(self, work_dir, inference_full_fname = 'full_inference.csv', inference_summary_fname = 'summary_inference.csv', 
                    confusion_matrix_fname = 'conf_matrix.png', summary_folder = "summary"):
        self.work_dir_summary = os.path.join(work_dir, summary_folder)
        self.inference_full_fname = inference_full_fname
        self.inference_summary_fname = inference_summary_fname
        self.inference_full_fpath = os.path.join(self.work_dir_summary, inference_full_fname)
        self.inference_summary_fpath = os.path.join(self.work_dir_summary, inference_summary_fname)
        self.confusion_matrix_fpath = os.path.join(self.work_dir_summary, confusion_matrix_fname)

        # Remove and initialize files
        self.remove_old_files()
        self.inference_full_file, self.inference_summary_file = self.init_sum_files()

        # Read the label list file, to minimize io processes 
        self.label_list = self.init_label_list()

    def init_label_list(self):
        '''
            Read the text file that contains all of the class labels into a list, such that one can easily access them throughout the program
        '''
        label_text_file_line_line_end = get_label_text_file()
        label_text_file = []
        for line in label_text_file_line_line_end:
            label_text_file.append(line.rstrip())
        return label_text_file

    def init_sum_files(self):
        '''
            Initialize the summary files
        '''
        inference_full_columns = ['File name', 'Actual Label' , 'Predicted Label', 'Predicted Values %']
        inference_summary_columns = ['Correct', 'Incorrect', 'Sum']

        inf_full = pd.DataFrame(columns = inference_full_columns)
        inf_sum = pd.DataFrame(columns = inference_summary_columns)  
        return  inf_full, inf_sum

    def remove_old_files(self):
        '''
            Remove the two generated files, such that one does not risk poluting new data with old data
        '''
        if os.path.isfile(self.inference_full_fpath):
            os.remove(self.inference_full_fpath)
        if os.path.isfile(self.inference_summary_fpath):
            os.remove(self.inference_summary_fpath)

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        '''
            Prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
        '''
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(self.confusion_matrix_fpath)

    def make_confusion_matrix(self):
        '''

        '''
        y_true = list(self.inference_full_file['Actual Label'])
        y_pred = list(self.inference_full_file['Predicted Label'])

        # Need the index of the labels for the confusion matrix
        for counter, _ in enumerate(y_true):
            y_true[counter] = self.label_list.index(y_true[counter])
            y_pred[counter] = self.label_list.index(y_pred[counter])

        conf_matrix = confusion_matrix(y_true, y_pred)
        
        class_names = self.label_list

        plt.figure()
        self.plot_confusion_matrix(conf_matrix, classes=class_names)

    def summarize_inference_full(self):
        summarized_dict = self.get_summarized_inference_dict()
        for k, v in summarized_dict.items():
            self.inference_summary_file.loc[k] = v
        self.inference_summary_file.to_csv(self.inference_summary_fpath)
        
    def get_summarized_inference_dict(self):
        class_name_list_unique = list(self.inference_full_file['Actual Label'].unique())
        sum_dict = dict.fromkeys(class_name_list_unique, {})

        gen_sum_dict = {'Correct': 0, 'Incorrect': 0, 'Sum': 0}

        for k, v in sum_dict.items():
            sum_dict[k] = copy.deepcopy(gen_sum_dict)
        
        for _, row in self.inference_full_file.iterrows():
            file_name = row['File name']
            y_true = row['Actual Label']
            y_pred = row['Predicted Label']
            if compare_strings(y_pred, y_true):
                sum_dict[y_true]['Correct'] += 1
            else:
                sum_dict[y_true]['Incorrect'] += 1
            sum_dict[y_true]['Sum'] += 1
        return sum_dict

    def inference_full_add_row(self, file_name, label, predicted_vals):
        '''
           Add a row to the 'full summary' (.csv) file 
        '''
        for counter, _ in enumerate(file_name):
            # Unpack the inputed values (Tensor) to a list
            predicted_values_list = [v.item() for v in predicted_vals[counter]]
            # Get the predictions in percentages
            preds_percentages = self.get_predictions_in_percentages(predicted_values_list)
            # Get the predicted label, and its value, by extracting the key with the highest value
            value, key = dict_max_value(dic = preds_percentages)
            # Easier to understand the class name, than the label index
            actual_label = self.label_list[label[counter].item()]

            new_row = [file_name[counter], actual_label, key, preds_percentages]
            print("New row: {}".format(new_row))
            self.inference_full_file.loc[len(self.inference_full_file)] = new_row

        self.inference_full_file.to_csv(self.inference_full_fpath, index=False)

    def get_predictions_in_percentages(self, li):
        min_value = np.amin(li)
        preds_pos = li + (min_value ** 2) # So it is a positive number
        preds_soft = softmax(preds_pos)
        top5 = preds_soft.argsort()[-5:][::-1]
        
        zipped = {}
        counter = 0
        for el in top5:
            counter += 1
            label_name = self.label_list[el]
            #label_name = get_class_name_from_index(index = el)
            zipped[label_name] = round((preds_soft[el]*100), 3)

        return zipped
