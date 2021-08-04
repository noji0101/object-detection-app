"""Models"""

from typing import Dict

from configs.supported_model import SUPPORTED_MODEL

from model.ssd_model import SSDModel

def get_model(config: Dict, is_eval: bool) -> object:
    """Get Model Class"""
    model_name = config['model']['name']

    if model_name in SUPPORTED_MODEL['SSD']:
        if is_eval:
            return SSDModel(config=config)
        else:
            return SSDModel(config=config)