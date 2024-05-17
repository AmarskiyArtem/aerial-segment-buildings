import os
import torch
from models.deeplab.modeling.sync_batchnorm.replicate import patch_replication_callback

def load_model(model, checkpoint_path, is_cuda=False, gpu_ids=0):
    # Load state_dict, if any
    if is_cuda:
        model_checkpoint = torch.load(checkpoint_path)
    else:
        model_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load model onto GPUs
    if is_cuda:
        assert gpu_ids is not None
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()


        if model_checkpoint is not None:
            model.load_state_dict(model_checkpoint)
    elif model_checkpoint is not None:
        try:
            model.load_state_dict(model_checkpoint)
        except RuntimeError:
            # The model is currently on the CPU, and does not have DataParallel wrapper
            # Need to remove the "module." prefix from all keys in state_dict
            from collections import OrderedDict
            new_checkpoint = OrderedDict()
            for module_name, parameters in model_checkpoint.items():
                name = module_name[7:] 
                new_checkpoint[name] = parameters
            model_checkpoint = new_checkpoint
            model.load_state_dict(model_checkpoint)
        
    return model
