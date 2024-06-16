from src.models.rgbfootprint.models.deeplab.modeling.backbone import drn


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == "drn_c42":
        return drn.drn_c_42(BatchNorm)
    else:
        raise NotImplementedError
