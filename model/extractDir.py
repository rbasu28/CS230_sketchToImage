import os

def extract_folder_names(root_dir, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Walk through the directory
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Write each directory name to the file
            for dirname in dirnames:
                file.write(f"{dirname}\n")

# Specify the root directory and the output file
root_directory = 'C:/Users/rub/Desktop/Stanford/CS230/Project/Zero-Shot-Sketch-Based-Image-Retrieval-master/Zero-Shot-Sketch-Based-Image-Retrieval-master/Dataset/photo'
output_filename = 'C:/Users/rub/Desktop/Stanford/CS230/Project/Zero-Shot-Sketch-Based-Image-Retrieval-master/Zero-Shot-Sketch-Based-Image-Retrieval-master/Dataset/train_labels.txt'

# Extract folder names and write to the text file
extract_folder_names(root_directory, output_filename)