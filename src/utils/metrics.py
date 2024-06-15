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

import numpy as np


def calculate_metrics(pred, real):
    iou_score = iou(pred, real)
    p = precision(pred, real)
    r = recall(pred, real)
    f1 = f1_score(pred, real)
    return iou_score, p, r, f1


def iou(pred, real):
    intersection = np.logical_and(pred, real)
    union = np.logical_or(pred, real)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def precision(pred, real):
    intersection = np.logical_and(pred, real)
    if np.sum(pred) == 0 and np.sum(real) == 0:
        return 1
    elif np.sum(pred) == 0:
        return 0
    return np.sum(intersection) / np.sum(pred)


def recall(pred, real):
    intersection = np.logical_and(pred, real)
    if np.sum(real) == 0:
        if np.sum(pred) == 0:
            return 1
        else:
            return 0
    return np.sum(intersection) / np.sum(real)


def f1_score(pred, real):
    p = precision(pred, real)
    r = recall(pred, real)
    return 2 * p * r / (p + r + 1e-5)
