from tools.utils.relevant_classes import relevant_classes
from tools.utils.file_util import *
from tools.views.output_messages import *

import os
import argparse

class extract_relevant_classes():
        '''
                This class is only relevant if the user has downloaded the 'original' dataset from ST-GCN
        '''

        def __init__(self, 
                        classes,
                        data_folder_path = "data/Kinetics/kinetics-skeleton",
                        output_dict_name = "kinetics_label_reduced.json"):
                '''
                        classes: List - containing names of the desired classes 
                        output_folder_path: String - path to data folder
                        output_file_name: String - name of the .json file in which to store the skeleton information 
                '''
                self.classes = classes

                # Dictionary to store the relevant class names with their new respective label index, used to update label index when moving the files
                self.dict_label_index = dict.fromkeys(self.classes)
                self.data_folder_path = data_folder_path

                # Divide the data between train and validation data
                dict_output_pre_ext, dict_output_file_ext = output_dict_name.split(".")

                self.output_dict_name_train = '{}_{}.{}'.format(dict_output_pre_ext, 'train', dict_output_file_ext)
                self.output_dict_name_val = '{}_{}.{}'.format(dict_output_pre_ext, 'val', dict_output_file_ext)

        def fill_label_index_dict(self, label_dict):
                '''
                        Add the class names to label text file (label_name_reduced.txt), and fill the dictionary with their index (e.g. line number in text file)

                        label_dict: Dictionary - Dictionary where the keys are the new class names, and their values are empty 
                '''
                for class_name in label_dict:
                        if not label_dict[class_name]:
                                label_dict[class_name] = update_label_list(class_name)

        def start(self):
                '''
                        Start the process of extracting relevant classes from the original dataset such that one is able to train the model on these

                        This includes:
                                - identifying and moving the skeleton files from kinetics_train/ and kinetics_val/ 
                                - identifying and moving the dictionary entries from kinetics_train_label.json and kinetics_val_label.json
                '''
                # Output message to user
                start_extracting_relevant_classes_message()

                self.fill_label_index_dict(self.dict_label_index)
 
                relevant_classes_dict_train = relevant_classes.get_relevant_classes_dict(class_list = self.classes,
                                                                                                label_index_dict = self.dict_label_index,  
                                                                                                input_path = self.data_folder_path,
                                                                                                train_val = 'train')

                relevant_classes_dict_val = relevant_classes.get_relevant_classes_dict(class_list = self.classes, 
                                                                                                label_index_dict = self.dict_label_index, 
                                                                                                input_path = self.data_folder_path,
                                                                                                train_val = 'val')

                # Move the relevant skeleton files from the training folder 
                relevant_classes.copy_relevant_files(dict_ = relevant_classes_dict_train, train_val = 'train', data_path = self.data_folder_path)

                # Move the relevant skeleton files from the validation folder
                relevant_classes.copy_relevant_files(dict_ = relevant_classes_dict_val, train_val = 'val', data_path = self.data_folder_path)

                # Path to the folder in which to move the relevant skeleton files
                output_dict_train_f_path = os.path.join(self.data_folder_path, self.output_dict_name_train)
                output_dict_val_f_path = os.path.join(self.data_folder_path, self.output_dict_name_val)

                # Add the specified dictionary entries to a reduced dictionary
                dict2file(output_dict_train_f_path, relevant_classes_dict_train)
                dict2file(output_dict_val_f_path, relevant_classes_dict_val)


if __name__ == '__main__':
        parser = argparse.ArgumentParser(
                description='This is a program in which one can specify class names which to extract, such that the model can be trained on these rather than the entire data set. NOTE: This is only relevant if the user has downloaded the "original" dataset from ST-GCN')

        parser.add_argument('-C', '--classes', nargs='+', help='List of class names')

        # Read arguments
        args = parser.parse_args()
        
        # Verify that the inputed classes are valid
        valid_classes = verify_new_classes(args.classes)

        # Could've specified deafult classes, but do not really see the value in this
        if valid_classes:
                extract = extract_relevant_classes(classes = args.classes)
                extract.start()
        else:
                print('\n The inputed classes were invalid, please try again')
        
