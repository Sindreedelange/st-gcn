from .file_util import *
import os
import shutil
from tools.views.output_messages import copy_files_message, copy_files_progress_message

class relevant_classes():

    @staticmethod
    def get_relevant_classes_dict(class_list, label_index_dict, input_path, train_val, relevant_classes_dict=None):
        '''
            Based on provided list of classes, extract the relevant dictionary entries, where the 'label' key == one of the classes in the class list
            Furthermore, updated its label_index, such that it corresponds to its line number in 'label_name_reduced.txt'

            class_list: List - class names that one wishes to extract 
            train_val: String - Are the classes to be extracted from the training set, or the validation set. 
                Can either be "train" or "val"
            relevant_classes_dict: String - Path to already existing dictionary (.json file) which we are to append the new information to

            Returns: Dictionary - filename: 
                                    has_skeleton: Boolean
                                    label: String 
                                    label_index: Integer
        '''

        # Get either 'kinetics_train_label' or 'kinetics_val_label'
        dict_extract_name = 'kinetics_{}_label.json'.format(train_val)
        dict_extract_f_path = os.path.join(input_path, dict_extract_name)
        dict_extract = file2dict(dict_extract_f_path)
        
        if relevant_classes_dict is None:
            return_dict = {}
        else:
            return_dict = file2dict(relevant_classes_dict)

        
        for (outer_key, outer_value) in dict_extract.items():
            # Finner de elemente med 'label' == klassene i class_list
            label = outer_value['label']
            label_index = label_index_dict.get(label)
            # Lager en ny dictionary med disse elementene
            return_dict[outer_key] = outer_value
            return_dict[outer_key]['label_index'] = label_index

        # Returnerer nye dictionary
        return return_dict

    @staticmethod
    def copy_relevant_files(dict_, train_val, data_path):
        '''
            Moving files from:
                'kinetics_train/'
            To
                'kinetics_train_reduced/'
            
            The files to be moved are the outer keys in the inputed dictionary
        '''

        old_folder = 'kinetics_{}'.format(train_val)
        old_folder_f_path = os.path.join(data_path, old_folder)

        new_folder = 'kinetics_{}_reduced'.format(train_val)
        new_folder_f_path = os.path.join(data_path, new_folder)

        # Inform user that the program is now moving files
        copy_files_message(input_folder = old_folder, output_folder = new_folder)

        counter = 0
        total_num_files = len(os.listdir(old_folder_f_path))

        if not os.path.exists(new_folder_f_path):
            os.makedirs(new_folder_f_path)

        # for every key, i.e. skeleton filename, in inputed dictionary
        for (outer_key, outer_value) in dict_.items():

            # Print progress
            copy_files_progress_message(counter, total_num_files)

            # Each skeletonfile has extension .json, so necessary in order to compare the strings
            outer_key_f_name = outer_key + ".json"

            # If there is a file in the old folder with this name, and the new folder does not already containt it, i.e. avoid duplicates
            if outer_key_f_name in os.listdir(old_folder_f_path) and outer_key_f_name not in os.listdir(new_folder_f_path):
                file_to_be_copied = os.path.join(old_folder_f_path, outer_key_f_name)
                file_to_be_copied_dict = file2dict(file_to_be_copied)

                # Edit the label index such that it corresponds to the class names' label index in the new label text file
                label_index = outer_value["label_index"]
                file_to_be_copied_dict["label_index"] = label_index
                
                file_to_copy_to = os.path.join(new_folder_f_path, outer_key_f_name)
                dict2file(file_to_copy_to, file_to_be_copied_dict)

            # No duplicates
            elif outer_key_f_name in os.listdir(new_folder_f_path):
                duplicate_files_error_message(new_folder_f_path, outer_key_f_name)
                

            counter += 1
