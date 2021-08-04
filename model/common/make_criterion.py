"""Make Criterion"""
import torch.nn as nn

from utils.logger import get_logger
from model.criterion import MultiBoxLoss

LOG = get_logger(__name__)

def make_criterion(criterion_cfg: object, device) -> object:
    if criterion_cfg.type == 'MultiBoxLoss':
        LOG.info('\n Criterion: MultiBoxLoss')
        return MultiBoxLoss(jaccard_thresh=criterion_cfg.jaccard_thresh, neg_pos=criterion_cfg.neg_pos, device=device)
    else:
        raise NotImplementedError('This loss function is not supported.')
