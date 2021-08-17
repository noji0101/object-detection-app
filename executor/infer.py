import argparse

from executor.inferrer import Inferrer


def parser():
    parser = argparse.ArgumentParser('Object Detection Argument')
    parser.add_argument('--configfile', type=str, default='./configs/default.yaml', help='config file')
    args = parser.parse_args()
    return args

def infer(args, img_path):
    inferer = Inferrer(args.configfile)
    img = inferer.show(img_path)
    return img

def infer_img(img_path):
    args = parser()
    img = infer(args, img_path)
    return img