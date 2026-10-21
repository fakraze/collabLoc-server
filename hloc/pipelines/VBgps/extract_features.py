import argparse
from pathlib import Path
import logging

from hloc import extract_features

from RAFT.vbgps import extract_flow_features

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=Path, default='test',
                    help='Name of the scene under datasets dir, default: %(default)s')

parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

args = parser.parse_args()

dataset = Path('datasets', args.name)
images = dataset / 'db_images/'
outputs = Path('outputs', args.name)

feature_conf = extract_features.confs['netvlad']
feature_path = extract_features.main(feature_conf, images, outputs)


if args.model is not None:
    extract_flow_features(images, outputs, args)
else:
    logging.info('RAFT features are not extracted')
