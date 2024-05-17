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
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours