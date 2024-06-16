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

import cv2
import torch


def get_bboxes_from_mask(mask, mode="xyxy"):
    contours = get_contours_from_mask(mask)
    bboxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if mode == "xywh":
            bboxes.append((x, y, width, height))
        else:
            bboxes.append((x, y, x + width, y + height))
    bboxes = torch.tensor(bboxes)
    return bboxes


def get_contours_from_mask(mask):
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours
