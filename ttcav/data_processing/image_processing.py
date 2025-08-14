from torch.utils.data import Dataset
from PIL import Image
import os 

class ImageDataset(Dataset):
    def __init__(self, path, processor, paths_list=None):
        self.processor = processor

        if paths_list is not None: # If the input is a list of paths 
            self.image_paths = paths_list

        elif os.path.isdir(path): # If the input is a folder path
            self.image_paths = [
                os.path.join(path, fname)
                for fname in os.listdir(path)
                if fname.endswith(".jpg") or fname.endswith(".JPEG")
            ]
        elif os.path.isfile(path) and (path.endswith(".jpg") or path.endswith(".JPEG")): # If the input is a path to a single image 
            self.image_paths = [path]
        else:
            raise ValueError(f"Invalid input: {path} is neither a folder nor a .jpg / .JPEG file.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), img_path