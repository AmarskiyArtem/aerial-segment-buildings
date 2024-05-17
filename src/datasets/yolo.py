from utils import get_bboxes_from_mask

from pathlib import Path
from torch.utils.data import Dataset

import torch
import cv2

class YOLO_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images = sorted(list(self.data_dir.glob("images/*.jpg")))
        self.masks = sorted(list(self.data_dir.glob("masks/*.png")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (640, 640))
        mask = torch.tensor(mask)

        return image, mask