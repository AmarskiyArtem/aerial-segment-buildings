from src.models.rgbfootprint.drn_c42 import DRN_c42
#from src.models.SAM.sam import SAM
from src.models.YOLO.yolo import YOLOseg

from src.datasets.sam import SAM_dataset, SAM_collate_fn
from src.datasets.yolo import YOLO_dataset
from src.datasets.rgbfootprint import RGBfootprint_dataset

from torch.utils.data import DataLoader

from src.utils.evaluator import Evaluator

def eval(model_name, checkpoint_path, device, data_dir, batch_size, model_type=None):
    model_name = model_name.lower()
    if model_name == 'drn_c42':
        model = DRN_c42(checkpoint_path, device)
        dataset = RGBfootprint_dataset(data_dir)
    elif model_name == 'sam':
        if model_type is None:
            raise ValueError('model_type must be provided for SAM model')
        #model = SAM(model_type, checkpoint_path, device)
        dataset = SAM_dataset(data_dir)
    elif model_name == 'yolo':
        model = YOLOseg(checkpoint_path, device)
        dataset = YOLO_dataset(data_dir)
    else:
        raise NotImplementedError

    if model_name == 'sam':
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=SAM_collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluator = Evaluator(model, loader, device, False)
    metrics = evaluator.evaluate()
    return metrics

model_name = 'drn_c42'
checkpoint_path = '../best_miou_checkpoint2.pth.tar'
device = 'cpu'
data_dir = '../yolo/mini'
batch_size = 2

print(eval(model_name, checkpoint_path, device, data_dir, batch_size))