import os
import shutil


def prepare_data(source, target, count):
    """
    Prepare data by copying files from a source directory to a target directory.

    Parameters:
    - source (str): The source directory containing files like 0.jpg, 1.jpg, ...
    - target (str): The target directory where files will be copied.
    - count (int): The number of files to copy.
    """
    # Ensure the target directory exists
    os.makedirs(target, exist_ok=True)

    for i in range(count):
        # Construct the source file name
        file_name = f"{i}.jpg"
        source_file = os.path.join(source, file_name)

        # Construct the target file path
        target_file = os.path.join(target, file_name)

        # Copy the file if it exists
        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
        else:
            print(f"File {source_file} does not exist. Skipping.")

# Example usage:
prepare_data("/Users/minglirui/train_data/album_covers_512", "album_photos", 5000)
