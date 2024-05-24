from src.models.rgbfootprint.models.utils.loader import load_model
from src.models.rgbfootprint.models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback
from src.models.rgbfootprint.models.deeplab.modeling import deeplab 

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

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def forward(self, images):
        return self.model(images)