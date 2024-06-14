from pathlib import Path
from torch.utils.data import Dataset

import torchvision.transforms as tt
import cv2

class RGBfootprint_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images = sorted(list(self.data_dir.glob("images/*.png")))
        self.masks = sorted(list(self.data_dir.glob("masks/*.png")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = tt.functional.to_tensor(image)
        
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (224, 224))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = tt.functional.to_tensor(mask)

        return image, mask