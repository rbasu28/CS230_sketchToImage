
import os
import subprocess
import sys
import py7zr
import zipfile

# Set the path to your dataset directory
path_dataset = "C:/Users/rub/Desktop/Stanford/CS230/Project/Zero-Shot-Sketch-Based-Image-Retrieval-master/Zero-Shot-Sketch-Based-Image-Retrieval-master/Dataset"
python_exec = "C:/Users/rub/AppData/Local/miniconda3/envs/cs230-env/python.exe"


# Ensure the directory exists
if not os.path.exists(path_dataset):
    os.makedirs(path_dataset)

# Function to download a file from Google Drive
def download_file(file_id, destination):
    python_exec = sys.executable  # Use the current Python interpreter
    command = f"{python_exec} src/download_gdrive.py {file_id} {destination}"
    subprocess.run(command, shell=True, check=True)

# Function to extract a .7z file using py7zr
def extract_7z(file_path, destination):
    with py7zr.SevenZipFile(file_path, mode='r') as z:
        z.extractall(path=destination)

# Function to unzip a file using zipfile
def unzip_file(file_path, destination):
    with zipfile.ZipFile(file_path, 'r') as z:
        z.extractall(path=destination)

# Download the Sketchy dataset
print("Downloading the Sketchy dataset (it will take some time)")
download_file("0B7ISyeE8QtDdTjE1MG9Gcy1kSkE", f"{path_dataset}/Sketchy.7z")

# Extract the Sketchy dataset
print("Unzipping it...")
extract_7z(f"{path_dataset}/Sketchy.7z", path_dataset)

# Clean up
os.remove(f"{path_dataset}/Sketchy.7z")
os.remove(f"{path_dataset}/README.txt")
os.rename(f"{path_dataset}/256x256", f"{path_dataset}/Sketchy")

# Download the extended photos of the Sketchy dataset
print("Downloading the extended photos of Sketchy dataset (it will take some time)")
download_file("0B2U-hnwRkpRrdGZKTzkwbkEwVkk", f"{path_dataset}/Sketchy/extended_photo.zip")

# Extract the extended photos
print("Unzipping it...")
unzip_file(f"{path_dataset}/Sketchy/extended_photo.zip", f"{path_dataset}/Sketchy")

# Clean up
os.remove(f"{path_dataset}/Sketchy/extended_photo.zip")
os.rename(f"{path_dataset}/Sketchy/EXTEND_image_sketchy", f"{path_dataset}/Sketchy/extended_photo")

# Remove unwanted directories
unwanted_dirs = [
    "sketch/tx_000000000010", "sketch/tx_000000000110", "sketch/tx_000000001010",
    "sketch/tx_000000001110", "sketch/tx_000100000000", "photo/tx_000100000000"
]
for dir in unwanted_dirs:
    path = f"{path_dataset}/Sketchy/{dir}"
    if os.path.exists(path):
        os.removedirs(path)

# Rename directories to fix inconsistent naming
renames = [
    ("sketch/tx_000000000000/hot-air_balloon", "sketch/tx_000000000000/hot_air_balloon"),
    ("sketch/tx_000000000000/jack-o-lantern", "sketch/tx_000000000000/jack_o_lantern"),
    ("photo/tx_000000000000/hot-air_balloon", "photo/tx_000000000000/hot_air_balloon"),
    ("photo/tx_000000000000/jack-o-lantern", "photo/tx_000000000000/jack_o_lantern"),
    ("extended_photo/hot-air_balloon", "extended_photo/hot_air_balloon"),
    ("extended_photo/jack-o-lantern", "extended_photo/jack_o_lantern")
]
for old_name, new_name in renames:
    old_path = f"{path_dataset}/Sketchy/{old_name}"
    new_path = f"{path_dataset}/Sketchy/{new_name}"
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

print("Done")
print("Sketchy dataset is now ready to be used")