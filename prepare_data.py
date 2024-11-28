import argparse
import os
import shutil

def prepare_data(source, target, labels_file, is_train, is_photo):
    """
    Prepare data by copying files from a source directory (structured as label/file)
    to a target subdirectory while maintaining the original labels.

    Parameters:
    - source_photo (str): The source photo directory, organized as label/file.
    - source_sketch (str): The source skedirectory, organized as label/file.
    - target (str): The target directory where the files will be copied.
    - labels_file (str): A txt file, where each line is a text label.
    - prefix (str): A subdirectory inside the target where the files will be copied.
    """
    # Read the labels from the labels file
    # train_or_test = 'train' if is_train else 'test'
    photo_or_sketch = 'photos' if is_photo else 'sketches'
    with open(labels_file, 'r') as file:
        valid_labels = set(line.strip() for line in file)

    # Define the full path for the target subdirectory
    target_subdir = os.path.join(target, photo_or_sketch)

    print(f"Preparing dataset from {source} to {target_subdir}")
    print(f'Coping files with labels: {valid_labels}')
    # Create target subdirectory if it does not exist
    if not os.path.exists(target_subdir):
        os.makedirs(target_subdir)

    # Walk through the source directory
    file_count = 0
    for label in os.listdir(source):
        label_path = os.path.join(source, label)

        # Skip if label is not in the valid labels
        if label not in valid_labels or not os.path.isdir(label_path):
            continue

        # Process each file in the label directory
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)

            # Skip if not a file
            if not os.path.isfile(file_path):
                continue

            # Construct target file path
            target_label_dir = os.path.join(target_subdir, label)
            if not os.path.exists(target_label_dir):
                os.makedirs(target_label_dir)

            target_file_path = os.path.join(target_label_dir, filename)

            # Copy file
            shutil.copy(file_path, target_file_path)

            file_count += 1

    print(f"Copied {file_count} files, Data preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for SBIR')
    parser.add_argument('--source_photo',
                        help='Source photo files directory path. Directory should contain SketchDatabase dataset.',
                        required=False,
                        default='/home/ec2-user/train_data/256x256/photo/tx_000000000000')
    parser.add_argument('--source_sketch',
                        help='Source sketch files directory path. Directory should contain SketchDatabase dataset.',
                        required=False,
                        default='/home/ec2-user/train_data/256x256/sketch/tx_000000000000')
    parser.add_argument('--target',
                        help='Target directory path where the test/train data will copy to.',
                        required=False,
                        default='Dataset')
    parser.add_argument('--train_labels', help='train label file', required=False, default='Dataset/train_labels.txt')
    parser.add_argument('--test_labels', help='train label file', required=False, default='Dataset/test_labels.txt')

    args = parser.parse_args()
    prepare_data(args.source_photo, args.target, args.train_labels, is_train=True, is_photo=True)
    prepare_data(args.source_sketch, args.target, args.train_labels, is_train=True, is_photo=False)
    prepare_data(args.source_photo, args.target, args.test_labels, is_train=False, is_photo=True)
    prepare_data(args.source_sketch, args.target, args.test_labels, is_train=False, is_photo=False)
