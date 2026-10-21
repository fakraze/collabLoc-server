import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'


def read_file_lines(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    return lines


def load_image(imfile, dsize=None):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if dsize is not None:
        img = cv2.resize(img, dsize=dsize)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_db_cameras_txt(file_path):
    db_cameras = {}
    lines = read_file_lines(file_path)
    for line_idx, line in enumerate(lines):
        if line[0] == '#':
            continue
        
        tokens = line.replace('\n', '').split()

        db_cameras[int(tokens[0])] = {
            'width': int(tokens[2]),
            'height': int(tokens[3]),
        }
    return db_cameras


def load_db_images_txt(file_path, db_cameras):
    # Load poses from db
    db_images = {}
    lines = read_file_lines(file_path)
    for line_idx, line in enumerate(lines):
        if line[0] == '#':
            continue

        tokens = line.replace('\n', '').split()

        if line_idx % 2 == 0:
            image_name = tokens[9]

            # q = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])])
            # t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
            # pose = Pose(q, t, 'colmap')

            camera_id = int(tokens[8])
            db_images[image_name] = {
                'points2d': [],
                'camera_id': camera_id,
            }
        else:
            for col in range(0, len(tokens), 3):
                point3d_id = int(tokens[col + 2])
                if point3d_id != -1:
                    db_images[image_name]['points2d'].append(np.array([float(tokens[col]), float(tokens[col + 1])]))
    
    return db_images


def viz(img1, img2, flo, file_name, args):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([flo, img1, img2], axis=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(f'{args.out}/{file_name.split("/")[-1]}', img_flo[:, :, [2,1,0]])


def mark_points(img, coords: list, thickness=5, color=[255, 0, 0]):
    
    thick_minus = thickness // 2
    thick_plus = thickness - thick_minus

    for coord in coords:
        x, y = coord[0], coord[1]
        y_min, y_max = max(y - thick_minus, 0), min(y + thick_plus, img.shape[0] - 1)
        x_min, x_max = max(x - thick_minus, 0), min(x + thick_plus, img.shape[1] - 1)
        img[y_min:y_max, x_min:x_max] = np.tile(np.array(color), [y_max - y_min, x_max - x_min, 1])

    return img


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    test_images_path = os.path.join(args.path, 'test_images')

    db_cameras_txt = load_db_cameras_txt(os.path.join(args.path, 'model/cameras.txt'))
    db_images_txt = load_db_images_txt(os.path.join(args.path, 'model/images.txt'), db_cameras_txt)
    dsize = (640, 480)

    with torch.no_grad():
        db_image_path = glob.glob(os.path.join(test_images_path, 'db/db_*.png')) + \
                        glob.glob(os.path.join(test_images_path, 'db/db_*.jpg'))

        client_images_path = glob.glob(os.path.join(test_images_path, 'clients/*.png')) + \
                             glob.glob(os.path.join(test_images_path, 'clients/*.jpg'))

        assert len(db_image_path) == 1, 'Should have 1 image from database model'
        db_image_name = db_image_path[0].split("/")[-1]
        db_image = load_image(db_image_path[0], dsize=dsize)
        
        
        assert db_image_name in db_images_txt, 'db_image should be in images.txt'
        db_image_meta = db_images_txt[db_image_name]
        db_image_camera_id = db_image_meta['camera_id']
        w_scale = dsize[0] / db_cameras_txt[db_image_camera_id]['width']
        h_scale = dsize[1] / db_cameras_txt[db_image_camera_id]['height']

        client_images_path = sorted(client_images_path)

        for client_image_path in client_images_path:
            client_image = load_image(client_image_path, dsize=(db_image.shape[3], db_image.shape[2]))

            padder = InputPadder(client_image.shape)
            db_image, client_image = padder.pad(db_image, client_image)

            flow_low, flow_up = model(db_image, client_image, iters=20, test_mode=True)

            db_image = db_image[0].permute(1,2,0).cpu().numpy()
            client_image = client_image[0].permute(1,2,0).cpu().numpy()
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            
            # map flow to rgb image
            flow_visual = flow_viz.flow_to_image(flow)

            # for local_feat in db_image_meta['points2d']:
            #     local_feat_x = int(local_feat[0] * w_scale)
            #     local_feat_y = int(local_feat[1] * h_scale)
            #     db_image[local_feat_y, local_feat_x] = flow_visual[local_feat_y, local_feat_x]

            # for local_feat in db_image_meta['points2d'][22:23]:
            for local_feat in db_image_meta['points2d'][:30]:
                local_feat_x = int(local_feat[0] * w_scale)
                local_feat_y = int(local_feat[1] * h_scale) 
                f = flow[local_feat_y, local_feat_x]
                db_image = mark_points(db_image, [(local_feat_x, local_feat_y)], color=flow_visual[local_feat_y, local_feat_x])
                client_image = mark_points(client_image,
                [(local_feat_x + int(f[0]), local_feat_y + int(f[1]))],
                color=flow_visual[local_feat_y, local_feat_x])
                
                print(flow[local_feat_y, local_feat_x])

            out_img = np.concatenate([flow_visual, db_image, client_image], axis=1)

            cv2.imwrite(f'{args.out}/{client_image_path.split("/")[-1]}', out_img[:, :, [2,1,0]])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--out', default='out', help="path for output")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
