import os
import json
from PIL import Image
from torch.utils.data import Dataset

class DenseCaptionedDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.anno_path = os.path.join(data_path, "annotations")
        self.img_path = os.path.join(data_path, "photos")
        self.annotations, self.anno_list = self._load_annotations()

    def _load_annotations(self):
        """
        Load all annotation JSON files into a list of dictionaries.
        """
        annotation_files = [
            os.path.join(self.anno_path, f)
            for f in os.listdir(self.anno_path)
            if f.endswith('.json')
        ]
        annotations = []
        anno_list = [f for f in os.listdir(self.anno_path) if f.endswith('.json')]
        for file in annotation_files:
            with open(file, 'r') as f:
                annotations.append(json.load(f))
        return annotations, anno_list

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        anno_name = self.anno_list[index]

        base_caption = annotation.get('short_caption', None)
        extra_caption = annotation.get('extra_caption', None)
        if base_caption is None or extra_caption is None:
            raise ValueError("This annotation is lack of key:extra_caption.")
        image_name = annotation.get('image', None)

        img_path = os.path.join(self.img_path, image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        caption = base_caption + ' ' + extra_caption
        return image, caption, img_path, anno_name

def dci_custom_collate_fn(batch):
    images, captions, images_names, anno_names = zip(*batch)  # 分离图像和描述
    return images, list(captions), images_names, anno_names