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

import torch

from src.utils.metrics import calculate_metrics


class Evaluator:
    def __init__(self, model, loader, device, isSAM=False):
        self.model = model
        self.loader = loader
        self.device = device
        self.isSAM = isSAM

    def evaluate(self):
        self.model.eval()
        dataset_len = 0
        metrics = {"iou": 0, "dice": 0, "precision": 0, "recall": 0, "f1": 0}
        for data in self.loader:
            if self.isSAM:
                images, mask, bboxes = data
                images = images
                bboxes = bboxes
                output = self.model(images, bboxes)
            else:
                images, mask = data
                images = images.to(self.device)
                with torch.no_grad():
                    output = self.model(images)

            for i in range(len(output)):
                pred = output[i].cpu().numpy()
                real = mask[i].cpu().numpy()
                dataset_len += 1
                iou, precision, recall, f1 = calculate_metrics(pred, real)
                metrics["iou"] += iou
                metrics["precision"] += precision
                metrics["recall"] += recall
                metrics["f1"] += f1

        for key in metrics.keys():
            metrics[key] /= dataset_len
        return metrics
