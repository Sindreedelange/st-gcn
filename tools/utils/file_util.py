import numpy as np
import pickle
import json
import ruamel.yaml
import os

import sys

def update_num_classes_yaml(train_demo = ['train', 'demo']):
    '''
        When generating .npy and .pkl file, used for the training, verify that the number of classes in the data set corresponds to the various config files,
        such as 'st-gcn/config/st_gcn/kinetics-skeleton/train.yaml'
    '''
    yaml = ruamel.yaml.YAML()

    # Count the number of classes
    label_file = get_label_text_file()
    counter = 0
    for line in label_file:
        counter += 1
        
    for f in train_demo:
        yaml_f_path = "config/st_gcn/kinetics-skeleton/{}.yaml".format(f)
        print("Updating '{}' num classes".format(yaml_f_path))

        with open(yaml_f_path) as f:
            yaml_file = yaml.load(f)

        yaml_file['model_args']['num_class'] = counter

        with open(yaml_f_path, "w") as f:
            yaml.dump(yaml_file, f)
        
def check_duplicates(folder_path, file_name):
    '''
        Checks if file already exists in given folder

        folder_path: String - path to the folder in which one wishes to see if the file already exists
        file_name: String - name of the file one wishes to see if already exists

        Return: Boolean
            True: File exsists = duplicate
            False: File does not exsists = not a duplicate
    '''
    file_list = os.listdir(folder_path)
    return file_name in file_list

def update_label_list(label, text_path="resource/kinetics_skeleton/label_name_reduced.txt"):
    '''
        Check if any of the new labels need to be put in the label list

        If the label already exists in the list, return its line number, i.e. its label_index. 
        If the label does not exist in the list, add it to the end, and return that line number

        label: The name of the class that should be added to the text file
        text_path: Path to the text file, containing all the label names, that is to be updated

        return: label index of the class
    '''
    try:
        label_text_file = open(text_path, 'r+')
    except Exception as e:
        label_text_file = open(text_path, 'w+')

    label_exsists = False

    # Used to keep track of, and store each labels' label_index
    counter = 0

    # Check if the class label already exists
    for line in label_text_file:
        if compare_strings(line, label):
            label_exsists = True
            break
        counter += 1

    # New label, add it to the text file, and update train.yaml's number of classes
    if not label_exsists:
        label_text_file.write(label + '\n')
    #    file_util.update_yaml()
        
    label_text_file.close

    return counter

def verify_new_classes(class_list, 
                        original_label_text_file = "resource/kinetics_skeleton/label_name.txt", 
                        new_label_text_file = "resource/kinetics_skeleton/label_name_reduced.txt"):
    '''
        When trying to extract specific classes from original dataset, verify that they are actually valid class names, and that they do not already exist in the data set

        class_list: List - List of class names
        original_label_text_file: String - path to where the text file which should contain the class names
        new_label_text_file: String - path to the new text file, which should not already contain the class name

        Return: List - List of validated class names
    '''

    print("Verifying inputed class list")
    #First, check if they already exists in the new text file
    new = get_label_text_file(path=new_label_text_file)
    for class_name in class_list:
        # Return to the top of the fil
        new.seek(0, 0)
        for line in new:
            if compare_strings(class_name, line):
                print("\n Unfornunately '{}' already exists in the reduced data set, so removing it from the class list \n".format(class_name))
                class_list.remove(class_name)
    # Second, check if they are actually valid classes, according to the original dataset
    # However, this is only necessary if not all of the class names are now removed, due to them already being contained in the new text file
    verified_class_list = False
    if len(class_list) > 0:
        original = get_label_text_file(path=original_label_text_file)
        for class_name in class_list:
            original.seek(0, 0)
            verified_class_name = False
            for line in original:
                if compare_strings(class_name, line):
                    print("\n Found '{}' in the old text file: verified \n".format(class_name))
                    verified_class_name = True
                    # Found the class name, get next class name and start from the beginning of the text file
                    break
                    
            if not verified_class_name:
                print("\n Unfornunately '{}' cannot be found in the original data set, so removing it from the class list \n".format(class_name))
            
            else:
                verified_class_list = True
    
    return verified_class_list

def get_label_text_file(path="resource/kinetics_skeleton/label_name_reduced.txt"):
    '''
        Returns the text file containing the class names of the reduced data set, used to train the model on
    '''
    return open(path, 'r+')

def update_yaml(file_path='st-gcn/config/st_gcn/kinetics-skeleton/train.yaml'):
    '''
        ######################
        Potentially unecessary --> Updating the file when generating the .npy and .pkl files instead
        See 'update_num_classes_train_yaml'
        #####################
    '''
    yaml = ruamel.yaml.YAML()
    with open(file_path) as f:
        train_yaml = yaml.load(f)
    train_yaml['model_args']['num_class'] += 1
    with open(file_path, "w") as f:
        yaml.dump(train_yaml, f)

def file2dict(file_path):
    '''
        Load a local file - returned as a dictionary
        file_path: path to file which to load in as a dictionary
        return dictionary
    '''
    data = {}
    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
    except:
        message = "Not able to load the specified dictionary ({}), returning an empty one ".format(file_path)
        print(message)
        # print_message(message)
    return data

def dict2file(file_path, file, append_or_overwrite='w'):
    ''' 
        Store a dictionary as a json file
        file_path: String - path to where the file is to be stored
        file: Dictionary - file to be stored
        append_or_overwrite: choose to either append to an already existing file, or overwrite the file
            'w' = overwrite
            'a+' = append
    '''
    
    old_dictionary = {}
    # Check if the file already exists, in which case we should append to it
    print("Writing to dictionary: {}".format(file_path))
    if os.path.exists(file_path):
        old_dictionary = file2dict(file_path)
        
        for key, val in file.items():
            # No duplicates
            if key not in old_dictionary:
                old_dictionary[key] = val
            else:
                duplicate_files_error_message(file_path, key)
    else:
        old_dictionary = file
    
    with open(file_path, append_or_overwrite) as fp:
        # Indent = 4 makes the .json file readable --> winning
        json.dump(old_dictionary, fp, indent=4)
    
def compare_strings(a, b):
    '''
        Compare two strings
        return: True if contains the same, else False
    '''
    return [c for c in a if c.isalpha()] == [c for c in b if c.isalpha()]

def verify_directory(dir_path):
    '''
        Check if path is a directory, if not, make new directory
        dir_path: String - Path to directory
    '''
    if not os.path.isdir(dir_path):
        print('{} not a directory, but making it now \n'.format(dir_path))
        os.makedirs(dir_path)