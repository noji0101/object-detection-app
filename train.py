import argparse
import sys
sys.path.append('../')

from utils.load import load_yaml
from model import get_model


def parser():
    parser = argparse.ArgumentParser('Object Detection Argument')
    parser.add_argument('--configfile', type=str, default='./configs/default.yaml', help='config file')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    args = parser.parse_args()
    return args

def run(args):
    """Builds model, loads data, trains and evaluates"""
    config = load_yaml(args.configfile)

    model = get_model(config, args.eval)
    model.load_data(args.eval)
    model.build(args.eval)
    
    if args.eval:
        model.evaluate()
    else:
        model.train()

if __name__ == '__main__':
    args = parser()
    run(args)