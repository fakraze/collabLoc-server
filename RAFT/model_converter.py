import sys
sys.path.append('core')

import argparse
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def convert(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    torch.save(model.state_dict(), args.model.replace('.', '-old.'), _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="model to convert")
    args = parser.parse_args()

    convert(args)
