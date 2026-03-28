import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, root, annotation_file=None, transform=None, idx=None, subset_size=2000, seed=42):
        self.root = root
        self.coco = COCO(annotation_file)
        self.transform = transform

        if idx is not None:
            self.image_ids = idx
        else:
            all_ids = list(self.coco.imgs.keys())
            
            if subset_size is not None:
                random.seed(seed)   
                if len(all_ids) > subset_size:
                    self.image_ids = random.sample(all_ids, subset_size)
                else:
                    self.image_ids = all_ids
            else:
                self.image_ids = all_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_filename = image_info['file_name']
        
        image_path = os.path.join(self.root, image_filename)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, image_filename, image_path
    
def custom_collate_fn(batch):
    images, image_filenames, ip = zip(*batch)
    return images, image_filenames, ip