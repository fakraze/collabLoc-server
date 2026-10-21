import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import logging
from tqdm import tqdm

from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder


DEVICE = 'cuda'
dshape = (480, 640, 3)
dsize = (dshape[1], dshape[0]) # (x, y)


def read_file_lines(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    return lines


def load_image(img, dsize=None):
    if dsize is not None:
        img = cv2.resize(img, dsize=dsize)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# def load_db_images_txt(file_path, db_cameras):
#     # Load poses from db
#     db_images = {}
#     lines = read_file_lines(file_path)
#     for line_idx, line in enumerate(lines):
#         if line[0] == '#':
#             continue

#         tokens = line.replace('\n', '').split()

#         if line_idx % 2 == 0:
#             image_name = tokens[9]

#             # q = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])])
#             # t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
#             # pose = Pose(q, t, 'colmap')

#             camera_id = int(tokens[8])
#             db_images[image_name] = {
#                 'points2d': [],
#                 'camera_id': camera_id,
#             }
#         else:
#             for col in range(0, len(tokens), 3):
#                 point3d_id = int(tokens[col + 2])
#                 if point3d_id != -1:
#                     db_images[image_name]['points2d'].append(np.array([float(tokens[col]), float(tokens[col + 1])]))
    
#     return db_images


# def viz(img1, img2, flo, file_name, args):
#     img1 = img1[0].permute(1,2,0).cpu().numpy()
#     img2 = img2[0].permute(1,2,0).cpu().numpy()
#     flo = flo[0].permute(1,2,0).cpu().numpy()
    
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([flo, img1, img2], axis=1)

#     cv2.imwrite(f'{args.out}/{file_name.split("/")[-1]}', img_flo[:, :, [2,1,0]])


# def mark_points(img, coords: list, thickness=5, color=[255, 0, 0]):
#     thick_minus = thickness // 2
#     thick_plus = thickness - thick_minus
#     for coord in coords:
#         x, y = coord[0], coord[1]
#         y_min, y_max = max(y - thick_minus, 0), min(y + thick_plus, img.shape[0] - 1)
#         x_min, x_max = max(x - thick_minus, 0), min(x + thick_plus, img.shape[1] - 1)
#         img[y_min:y_max, x_min:x_max] = np.tile(np.array(color), [y_max - y_min, x_max - x_min, 1])
#     return img


def get_flow_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model


def get_flow(pad_src, fmap_src, fmap_dst, model, iters, args):
    with torch.no_grad():
        flow_low, flow_up = model.fast_forward(pad_src, fmap_src, fmap_dst, iters=iters, test_mode=True)
        flow = flow_up[0].permute(1,2,0).cpu().numpy()
        return flow

        # Map flow to rgb image
        # flow_visual = flow_viz.flow_to_image(flow)
        # return flow, flow_visual


def get_flow_feature_and_pad(img, flow_model):
    with torch.no_grad():
        img = load_image(img)
        padder = InputPadder(img.shape)
        img, = padder.pad(img)

        fmap = flow_model.extract_feature(img)
        return fmap, img


def extract_flow_features(images_path, outputs_path, args):
    logging.info('Extracting RAFT features')
    feature_filename = 'feats-raft-small.npy' if args.small else 'feats-raft.npy'
    pad_filename = 'pads-raft-small.npy' if args.small else 'pads-raft.npy'

    with torch.no_grad():
        flow_features = {}
        flow_pads = {}
        flow_model = get_flow_model(args)

        for image_name in tqdm(os.listdir(images_path)):
            db_img = cv2.imread(str(images_path / image_name))
            old_shape = db_img.shape
            scaled_db_img = cv2.resize(db_img, dsize=dsize)
            fmap, pad_img = get_flow_feature_and_pad(scaled_db_img, flow_model)
            flow_features[image_name] = fmap
            flow_pads[image_name] = {
                'pad': pad_img,
                'old_shape': old_shape,
            }

        np.save(outputs_path / feature_filename, flow_features)
        np.save(outputs_path / pad_filename, flow_pads)
    logging.info('Finished extracting RAFT features.')
