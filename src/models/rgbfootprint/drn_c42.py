from models.utils.loader import load_model
from models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab.modeling import deeplab 

class DRN_c42():
    def __init__(self, checkpoint_path, device):
        self.model =  deeplab.DeepLab(num_classes=2,
                        backbone='drn_c42',
                        output_stride=8,
                        sync_bn=True,
                        freeze_bn=False,
                        dropout_low=0.3,
                        dropout_high=0.5,
                    )
        self.model = load_model(self.model, checkpoint_path, device)
        

    def forward(self, images):
        return self.model(images)