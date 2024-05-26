from src.models.rgbfootprint.models.utils.loader import load_model
from src.models.rgbfootprint.models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback
from src.models.rgbfootprint.models.deeplab.modeling import deeplab 

import torch

class DRN_c42():
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model =  deeplab.DeepLab(num_classes=2,
                        backbone='drn_c42',
                        output_stride=8,
                        sync_bn=True,
                        freeze_bn=False,
                        dropout_low=0.3,
                        dropout_high=0.5,
                    )
        cuda = device == 'cuda'
        self.model = load_model(self.model, checkpoint_path, cuda)
        print('Drn_c42 loaded')

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, images):
        outputs = self.model(images)
        output_masks = [0] * images.shape[0]
        for i in range(outputs.shape[0]):
            pred = torch.nn.functional.softmax(outputs[i], dim=0)
            pred = torch.argmax(pred, axis=0)
            output_masks[i] = pred
        return output_masks
    
    def eval(self):
        self.model.eval()