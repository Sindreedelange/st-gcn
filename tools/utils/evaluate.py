import os
import pandas as pd
import numpy as np
from scipy.special import softmax
import copy
from .file_util import dict_max_value, get_class_name_from_index, compare_strings

class Evaluate():

    def __init__(self, work_dir, inference_full_fname = 'full_inference.csv', inference_summary_fname = 'summary_inference.csv', summary_folder = "summary"):
        self.work_dir_summary = os.path.join(work_dir, summary_folder)
        self.inference_full_fname = inference_full_fname
        self.inference_summary_fname = inference_summary_fname
        self.inference_full_fpath = os.path.join(self.work_dir_summary, inference_full_fname)
        self.inference_summary_fpath = os.path.join(self.work_dir_summary, inference_summary_fname)

        # Remove and initialize files
        self.remove_old_files()
        self.inference_full_file, self.inference_summary_file = self.initialize_files()

    def initialize_files(self):
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


    def get_inference_full_file(self):
        return self.inference_full_file

    def get_inference_summary_file(self):
        return self.inference_summary_file

    def inference_full_add_row(self, file_name, label, predicted_vals):
        '''
            
        '''
        for counter, _ in enumerate(file_name):
            # Unpack the inputed values (Tensor) to a list
            predicted_values_list = [v.item() for v in predicted_vals[counter]]
            preds_percentages = self.get_predictions_in_percentages(predicted_values_list)
            key, value = dict_max_value(dic = preds_percentages)
            actual_label = get_class_name_from_index(label[counter].item())

            new_row = [file_name[counter], actual_label, key, preds_percentages]
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
            label_name = get_class_name_from_index(index = el)
            zipped[label_name] = round((preds_soft[el]*100), 3)

        return zipped
