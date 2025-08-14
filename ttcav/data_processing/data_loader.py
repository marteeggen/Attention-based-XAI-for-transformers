from torch.utils.data import DataLoader
import os
from ttcav.data_processing.image_processing import ImageDataset
import random

def get_dataloader(processor, path, batch_size): 
    dataset = ImageDataset(path, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader 

def get_dataloader_sampling_multiple_folders(processor, folders, num_images, batch_size, seed=None):
    
    all_image_paths = []
    for folder in folders:
        if not os.path.isdir(folder):
            raise ValueError(f"Invalid folder path: {folder}")
        image_paths = [
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.endswith(".jpg") or fname.endswith(".JPEG")
        ]
        all_image_paths.extend(image_paths)

    if len(all_image_paths) < num_images:
        raise ValueError(f"Requested {num_images} images, but only found {len(all_image_paths)} images.")

    rng = random.Random(seed)
    sampled_paths = rng.sample(all_image_paths, num_images)
    
    dataset = ImageDataset(path=None, processor=processor, paths_list=sampled_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
