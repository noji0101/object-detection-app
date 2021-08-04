import argparse

from executor.inferrer import Inferrer, SSDPredictShow


def parser():
    parser = argparse.ArgumentParser('Object Detection Argument')
    parser.add_argument('--configfile', type=str, default='./configs/default.yaml', help='config file')
    args = parser.parse_args()
    return args

def infer(args):
    img_path = 'test_input.jpg'
    print(1)
    inferer = SSDPredictShow(args.configfile)
    inferer.show(img_path, data_confidence_level=0.5)

if __name__ == '__main__':
    args = parser()
    infer(args)