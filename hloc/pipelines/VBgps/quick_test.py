from collections import defaultdict
from distutils.log import info
from pathlib import Path
from pprint import pformat
import argparse
import time
import cv2
from cv2 import log
import h5py
import torch
import logging
import select
import numpy as np 
import os
import yaml

from hloc import extractors
from hloc.utils.read_write_model import read_model, qvec2rotmat
from hloc.utils.io import list_h5_names
from hloc.utils.base_model import dynamic_load
from hloc.utils.vbgps import get_fused_pose_from_arpose_str, get_poses_from_model, pairs_from_img_pose, match_features_from_flow, load_flow_features_and_pads

from RAFT.vbgps import get_flow_model

from ... import extract_features, match_features
from ... import pairs_from_covisibility, pairs_from_retrieval
from ... import colmap_from_nvm, triangulation, localize_sfm


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, default='config/test.yaml',
                    help='Path to the configuration file, default: %(default)s')
parser.add_argument('--test_images', type=Path,
                    help='Path to the test images dir')
parser.add_argument('--out_poses', type=Path, default='poses.txt',
                    help='Path to the output poses txt, default: %(default)s')
# parser.add_argument('--dataset', type=Path, default='datasets/test',
#                     help='Path to the dataset, default: %(default)s')
# parser.add_argument('--outputs', type=Path, default='outputs/test',
#                     help='Path to the output directory, default: %(default)s')
# parser.add_argument('--num_loc', type=int, default=20,
#                     help='Number of image pairs for loc, default: %(default)s')

parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--out', default='out', help="path for output")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

dataset = Path(config.get('dataset', 'dataset/test'))
outputs = Path(config.get('outputs', 'outputs/test'))
num_loc = int(config.get('num_loc', 20))
# num_loc = int(config.get('num_loc', 3))
# num_loc = int(config.get('num_loc', 1))
scale = float(config.get('scale', 1))
print('Configs')
print({'dataset': dataset, 'outputs': outputs, 'num_loc': num_loc, 'scale': scale})

# Setup the paths
images = dataset / 'images_upright/'
images.mkdir(exist_ok=True, parents=True)

reference_sfm = outputs / 'sfm_superpoint+NN'  # the SfM model we will build
db_global_feats = outputs / f'global-feats-netvlad.h5'
db_feats = outputs / f'feats-superpoint-n4096-r1024.h5'
loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'  # top-k retrieved by NetVLAD
results = outputs / f'Aachen_hloc_superpoint+superglue_netvlad{num_loc}.txt'

outputs_query_data = outputs / 'query_data/'
outputs_query_data.mkdir(exist_ok = True, parents=True)

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['NN-superpoint']
intrinsic = dataset / 'queries' / 'intrinsic.txt'


def load_db_gfeats(path, key='global_descriptor'):
    db_names = list_h5_names(path)
    with h5py.File(str(path), 'r') as fd:
        desc = [fd[n][key].__array__() for n in db_names]
    return desc, db_names
def load_db_localfeats(path):
    path = [path]
    name2ref = {n: i for i, p in enumerate(path)
                for n in list_h5_names(p)}
    return name2ref
def load_netvlad():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, retrieval_conf['model']['name'])
    model = Model(retrieval_conf['model']).eval().to(device)
    return model

def parse_intrinsic(path):
    with open(path, 'r') as f:
        for line in f:
            line.strip('\n')
            data = line.split(' ')
            camera_model, width, height, *params = data
            params = np.array(params, float)
            info = (camera_model, int(width), int(height), params)
    return info

def pose_postproc(qvec ,tvec):
    R_t = np.eye(4)
    R = qvec2rotmat(qvec)
    R_t[:3, :3] = R.T
    R_t[:3, 3] = -R.T @ tvec
    result = np.array(R_t).reshape(-1,1)
    result = np.array(['{:+.11f}'.format(result[n][0]) for n in range(len(result))], dtype='a11').reshape(-1, 1)
    result = np.array2string(result)

    return result

def recv_info(sock, count):
    data = b''
    while count:
        # recv(count) arg 'count' is maximum size 
        buf = sock.recv(count)
        if not buf : return None
        data += buf
        count -= len(buf)
    return data

logging.info('load SfM model.')
sfm_model = read_model(str(reference_sfm))
sfm_model_poses = get_poses_from_model(sfm_model)
logging.info('load db global feats.')
db_gdesc, db_names = load_db_gfeats(db_global_feats)
logging.info('load NetVlad model.')
netvlad = load_netvlad()
logging.info('load db local feats.')
db_local_name2ref = load_db_localfeats(db_feats)
query_info = parse_intrinsic(intrinsic)

logging.info('prepare flow model.')
flow_model = get_flow_model(args)
logging.info('load flow feats.')
flow_features, flow_pads = load_flow_features_and_pads(outputs, args)


for i in range(1):
    with open(args.out_poses, 'w') as f:
        f.write(f'# img_name [ qw qx qy qz] [ x y z]\n')

        for frame_idx, my_img_name in enumerate(os.listdir(args.test_images), 1):
            if '.txt' in my_img_name:
                continue
            
            t0 = time.monotonic()
            
            # Remove old data
            for old_img_name in os.listdir(images):
                os.unlink(images / old_img_name)
            for old_item in os.listdir(outputs_query_data):
                os.unlink(outputs_query_data / old_item)
            for old_item in os.listdir(outputs):
                if '.txt' in old_item:
                    # Remove all .txt items except pairs-exhaustive.txt
                    if old_item == 'pairs-exhaustive.txt':
                        pass
                    else:
                        os.unlink(outputs / old_item)
                elif '.7_pairs-query-netvlad' in old_item:
                    # Remove all .7_pairs-query-netvlad20 items
                    os.unlink(outputs / old_item)

            t1 = time.monotonic()
            
            image_retrieval_by = 'global_feature'
            pose_file_path = str(args.test_images / os.path.splitext(my_img_name)[0]) + '.txt'
            if os.path.exists(pose_file_path):
                with open(pose_file_path, 'r') as pose_file:
                    pose_str = pose_file.readline()
                    cur_fused_pose = get_fused_pose_from_arpose_str(pose_str)
                    print('cur_fused_pose', cur_fused_pose.to_formatted_str('%qw %qx %qy %qz %tx %ty %tz'))
                    image_retrieval_by = 'fused_pose'

            feature_matching_by = 'optical_flow'
            feature_matching_by = 'superpoint+NN'

            t2 = time.monotonic()

            query_img = cv2.imread(str(args.test_images / my_img_name), cv2.IMREAD_COLOR)
            query_img = cv2.resize(query_img, (640, 480))
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            print(query_img.shape)
            img_name = f"q_{frame_idx}.jpg"
            img_path = Path(images, img_name)
            cv2.imwrite(str(img_path), query_img)

            feat_output = Path(outputs, "query_data")
            features = extract_features.main(feature_conf, images, feat_output)

            t3 = time.monotonic()

            print('image_retrieval_by', image_retrieval_by)
            if image_retrieval_by == 'fused_pose':
                # Retrieve image by fused pose
                pairs_from_img_pose(img_name, sfm_model_poses, cur_fused_pose, loc_pairs, num_loc)
            else:
                # Retrieve image by global feature
                global_descriptors = extract_features.main(retrieval_conf, images, feat_output, model=netvlad)
                pairs_from_retrieval.main(
                    global_descriptors, loc_pairs, num_loc,
                    query_prefix='', db_prefix='', db_descriptors=db_gdesc, db_names=db_names)

            t4 = time.monotonic()

            if feature_matching_by == 'optical_flow':
                loc_matches = match_features_from_flow(sfm_model, flow_features, loc_pairs, features, flow_pads,
                                                    matches_path=Path(outputs, f"{features.stem}_{matcher_conf['output']}_{loc_pairs.stem}"),
                                                    query_img=query_img, flow_model=flow_model, args=args)
            else:
                loc_matches = match_features.main(matcher_conf, loc_pairs, features, matches=
                                                Path(outputs, f"{features.stem}_{matcher_conf['output']}_{loc_pairs.stem}"), 
                                                features_ref=db_feats, name2ref = db_local_name2ref)
            t5 = time.monotonic()

            poses = localize_sfm.main(
                sfm_model,
                #dataset / 'queries/*_time_queries_with_intrinsics.txt',
                [(img_name, query_info)],
                loc_pairs,
                features,
                loc_matches,
                results,
                covisibility_clustering=True)  # not required with SuperPoint+SuperGlue

            t6 = time.monotonic()
            #send pose
            qvec, tvec = poses[img_name]

            #remove img and feature data of query 
            features.unlink()
            if image_retrieval_by == 'global_feature':
                global_descriptors.unlink()
            img_path.unlink()

            t7 = time.monotonic()

            f.write(f'{my_img_name} {qvec} {tvec}\n')
            print(f'{my_img_name} {qvec} {tvec}'.replace('[', '').replace(']', ''))

            print(f'Initialize: {round(t1 - t0, 3)}', end='\t')
            print(f'Local feat: {round(t3 - t2, 3)}', end='\t')
            print(f'Retrieval ({image_retrieval_by}): {round(t4 - t3, 3)}', end='\t')
            print(f'Matching ({feature_matching_by}): {round(t5 - t4, 3)}', end='\t')
            print(f'Localize: {round(t6 - t5, 3)}', end='\t')
            print(f'All: {round(t7 - t0, 3)}')

        print(f'Poses written to {args.out_poses}')
