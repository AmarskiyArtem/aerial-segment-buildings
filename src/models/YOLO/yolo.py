# Copyright (c) 2024, Artem Amarskiy, Anastasiia Kornilova
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ultralytics import YOLO

import numpy as np
import torch


class YOLOseg:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = YOLO(checkpoint_path).to(device)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, images):
        outputs = self.model(images)
        output_masks = [0] * images.shape[0]
        for i in range(len(outputs)):
            if outputs[i].masks:
                masks = outputs[i].masks.data
                mask = torch.any(masks, dim=0).squeeze()
            else:
                mask = torch.zeros(images.shape[2], images.shape[3])
            output_masks[i] = mask
        return output_masks

    def eval(self):
        pass
