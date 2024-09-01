import os 

def generate_negative_description_file():
    #Open the output file for writing. Will overwrite all existing data in there
    with open('neg.txt', 'w') as f:
        #Loop over all files
        for filename in os.listdir('negative'):
            f.write('negative/' + filename + '\n')
            
            
def remove_whitespace_in_filenames(folder_path):
    # List all files in the directory
    for filename in os.listdir(folder_path):
        # Skip directories, only rename files
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Create the new filename by replacing spaces with nothing
            new_filename = filename.replace(" ", "")
            # Get the full path for both old and new file names
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')