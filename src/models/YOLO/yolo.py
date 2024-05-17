from ultralytics import YOLO

import numpy as np
import torch

class YOLO:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = YOLO(checkpoint_path).to(device)
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def forward(self, images):
        outputs = self.model(images)
        output_masks = [0] * images.shape[0]
        for i in range(len(outputs)):
            masks = outputs[i].masks.data
            mask = torch.any(masks, dim=0).squeeze()
            output_masks[i] = mask
        return output_masks