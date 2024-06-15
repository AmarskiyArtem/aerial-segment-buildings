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

from src.datasets.utils import get_bboxes_from_mask

from pathlib import Path
from torch.utils.data import Dataset

import cv2


class SAM_dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.images = sorted(list(self.data_dir.glob("images/*.png")))
        self.masks = sorted(list(self.data_dir.glob("masks/*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bboxes = get_bboxes_from_mask(mask)

        return image, mask / 255, bboxes


def SAM_collate_fn(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]

    return images, masks, bboxes
