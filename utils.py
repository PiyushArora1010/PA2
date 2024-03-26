import torch

from module.models import ECAPA_TDNN_SMALL

def load_checkpoint(model, checkpoint):
    model = ECAPA_TDNN_SMALL(**model)
    if checkpoint != 'none':
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model