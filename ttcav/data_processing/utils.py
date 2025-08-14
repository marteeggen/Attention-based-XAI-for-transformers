import os
import shutil
import random
import re
import torch
from collections import defaultdict
import numpy as np

torch.manual_seed(1337)
random.seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            file_paths.append(full_path)
    return file_paths

def convert(obj):
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def copy_images(source_folder, destination_folder, concept):
    """
    Copy .jpg images from a source folder to another folder.
    """
    os.makedirs(destination_folder, exist_ok=False)
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".jpg") and file_name.startswith(concept):
            shutil.copy(os.path.join(source_folder, file_name), os.path.join(destination_folder, file_name))

def sample_images(source_folder, destination_folder, num_images):
    """
    Copy N randomly selected images from a source folder to another folder. 
    """
    os.makedirs(destination_folder, exist_ok=False)
    image_files = [file_name for file_name in os.listdir(source_folder) if (file_name.endswith(('.jpg')) or file_name.endswith(('.JPEG')))]
    num_images = min(num_images, len(image_files))
    random_files = random.sample(image_files, num_images)
    for file_name in random_files:
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(destination_folder, file_name))

def get_num_items_in_folder(folder_path):
    """
    Return the number of items or files in a folder. 
    """
    all_items = os.listdir(folder_path)
    file_count = sum(1 for item in all_items if os.path.isfile(os.path.join(folder_path, item)))
    return file_count

def filter_dataset_ade20k(source_folder_path, destination_folder_path):
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    pattern = re.compile(r'^ADE_train_\d{5,}\.(jpg|jpeg|png|bmp|tif|tiff)$', re.IGNORECASE)

    for filename in os.listdir(source_folder_path):
        if pattern.match(filename):
            src_path = os.path.join(source_folder_path, filename)
            dst_path = os.path.join(destination_folder_path, filename)
            shutil.copy2(src_path, dst_path)



