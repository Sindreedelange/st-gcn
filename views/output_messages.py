def print_generic_message(message):
    print("\n ------------------------- \n", message, "\n ------------------------- \n")

def moving_files_message(input_folder, output_folder):
    print("Moving skeleton files from: {}, to {} \n".format(input_folder, output_folder))

def start_extracting_relevant_classes_message():
    print("Starting extracting relevant classes \n")

def duplicate_files_error_message(output_folder, file_name):
    '''
        Inform the user that the files already exists
        output_folder: String - Folder in which to look for the file
        file_name: String - Name of the file to look for
    '''
    print("Duplicate error: {} already contains {}".format(output_folder, file_name))

def open_file_error_message(exception, text_path):
    '''
        Inform the user that the program was unable to open the desired text file

        exception: Exception 
        text_path: String 
    '''
    print("Error message: {} \n".format(exception))
    print("Unable to open {}, because file does not exists (or you might not have access?) \n Exiting program \n".format(text_path))

def moving_files_progress_message(n, total):
    '''
        Inform the user about the progress made of the files that are being moved.
        Works kind of like a text based progress bar
        n: Integer - current file number being moved
        total: Integer - total number of files that are being moved
    '''
    print("Files moved: {}/{}".format(n, total), end='\r')


     