import numpy as np

def calculate_metrics(pred, real):
    iou_score = iou(pred, real)
    dice = dice_coefficient(pred, real)
    p = precision(pred, real)
    r = recall(pred, real)
    f1 = f1_score(pred, real)
    return iou_score, dice, p, r, f1

def iou(pred, real):
    intersection = np.logical_and(pred, real)
    union = np.logical_or(pred, real)
    if np.sum(union) == 0:
        return 1
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def dice_coefficient(pred, real):
    intersection = np.logical_and(pred, real)
    if np.sum(pred) + np.sum(real) == 0:
        return 1
    return 2. * np.sum(intersection) / (np.sum(pred) + np.sum(real))

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
        return 1
    return np.sum(intersection) / np.sum(real)

def f1_score(pred, real):
    p = precision(pred, real)
    r = recall(pred, real)
    return 2 * p * r / (p + r + 1e-5)