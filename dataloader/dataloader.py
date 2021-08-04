"""DataLoader class"""
from typing import List

import torch.utils.data as data

from .utils import make_datapath_list, Anno_xml2list, od_collate_fn
from .dataset import VOCDataset
from .transform import DataTransform


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(dataroot: str):
        """load dataset from path
        Parameters
        ----------
        dataroot : str
            path to the image data directory e.g. './data/images/'
        labelpath : str
            path to the label csv e.g. './data/labels/train.csv'
        Returns
        -------
        Tuple of list
            img_list: e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
            lbl_list: e.g. [3, 5, ...]
        """
        train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(dataroot)
        return (train_img_list, train_anno_list, val_img_list, val_anno_list)

    @staticmethod
    def preprocess_data(data_config: object, img_list: List, anno_list: List, batch_size: int, mode: str):
        """Preprocess dataset
        Parameters
        ----------
        data_config : object
            data configuration
        img_list : List
            a list of image paths
        lbl_list : List
            a list of labels
        batch_size : int
            batch_size
        mode : str
            'train' or 'eval'
        Returns
        -------
        Object : 
            DataLoader instance
        Raises
        ------
        ValueError
            raise value error if the mode is not 'train' or 'eval'
        """
        # transform
        input_size = data_config.input_size
        color_mean = tuple(data_config.color_mean)
        transform = DataTransform(input_size, color_mean)
        transform_anno = Anno_xml2list(data_config.voc_classes)

        # dataset
        dataset = VOCDataset(img_list, anno_list, mode, transform, transform_anno)

        # dataloader
        if mode == 'train':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=True ,collate_fn=od_collate_fn)
        elif mode == 'eval':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=False ,collate_fn=od_collate_fn)
        else:
            raise ValueError('the mode should be train or eval. this mode is not supported')