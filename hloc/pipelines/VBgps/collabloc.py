import os
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
import socket
import select
import numpy as np 
from datetime import datetime
import threading
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
scale = float(config.get('scale', 1))
preload_flow_feats = bool(config.get('preload_flow_feats', True))
print('Configs')
print({'dataset': dataset, 'outputs': outputs, 'num_loc': num_loc, 'scale': scale, 'preload_flow_feats': preload_flow_feats})

# Setup the paths
images = dataset / 'images_upright/'
images.mkdir(exist_ok = True, parents=True)

recv_images = dataset / 'recv_images/'
recv_images.mkdir(exist_ok = True, parents=True)

db_images = dataset / 'db_images/'

reference_sfm = outputs / 'sfm_superpoint+NN'  # the SfM model we will build
db_global_feats = outputs / f'global-feats-netvlad.h5'
db_feats = outputs / f'feats-superpoint-n4096-r1024.h5'
loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'  # top-k retrieved by NetVLAD
results = outputs / f'Aachen_hloc_superpoint+superglue_netvlad{num_loc}.txt'

outputs_query_data = outputs / 'query_data/'
outputs_query_data.mkdir(exist_ok = True, parents=True)

for old_img_name in os.listdir(images):
    os.unlink(images / old_img_name)
for old_img_name in os.listdir(recv_images):
    os.unlink(recv_images / old_img_name)
for old_item in os.listdir(outputs_query_data):
    os.unlink(outputs_query_data / old_item)
for old_item in os.listdir(outputs):
    if '.txt' in old_item:
        if old_item == 'pairs-exhaustive.txt':
            pass
        else:
            os.unlink(outputs / old_item)
    elif '.7_pairs-query-netvlad' in old_item:
        os.unlink(outputs / old_item)

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['NN-superpoint']


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

def arcore_parse_intrinsic(image_shape, focal_length):
    line = f'PINHOLE {image_shape[1]} {image_shape[0]} {focal_length} {image_shape[1] / 2} {image_shape[0] / 2}'
    data = line.split()
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
sfm_model_poses = get_poses_from_model(sfm_model, db_images)
logging.info('load db global feats.')
db_gdesc, db_names = load_db_gfeats(db_global_feats)
logging.info('load NetVlad model.')
netvlad = load_netvlad()
logging.info('load db local feats.')
db_local_name2ref = load_db_localfeats(db_feats)

logging.info('prepare flow model.')
flow_model = get_flow_model(args)
if preload_flow_feats:
    logging.info('load flow feats.')
    flow_features, flow_pads = load_flow_features_and_pads(outputs, args)
    scaled_db_imgs = {}
else:
    logging.info('load db_images.')
    flow_features, flow_pads, scaled_db_imgs = {}, {}, {}
    dshape = (480, 640, 3)
    dsize = (dshape[1], dshape[0]) # (x, y)
    for image_name in os.listdir(db_images):
        db_img = cv2.imread(str(db_images / image_name))
        old_shape = db_img.shape
        scaled_db_img = cv2.resize(db_img, dsize=dsize)
        scaled_db_imgs[image_name] = {
            'old_shape': old_shape,
            'img': scaled_db_img
        }

min_track_conf_lock = threading.Lock()
is_reading_query_image_lock = threading.Lock()


def client_connection_thread():
    global min_track_conf
    global next_serve_sock
    global cur_serve_sock
    global is_reading_query_image

    ssock = socket.socket()
    addr = ('140.113.195.250', 9999)
    ssock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ssock.setblocking(False)
    ssock.bind(addr)
    ssock.listen(5)
    fds = [ssock]

    while True:
        # Non-blocking select
        # logging.info("waiting for connection...")
        try:
            readsocks, _, _  = select.select(fds, [], [])
        except:
            # Remove closed sockets from fds
            logging.info("removing closed sockets...")
            for sock in fds:
                if sock.fileno() == -1:
                    fds.remove(sock)
            continue

        for sock in readsocks:
            try:
                # Start receiving from connection
                c = sock.fileno()
                with is_reading_query_image_lock:
                    if sock is ssock:
                        # New connection
                        logging.info(f"got new connection from {addr}")
                        csock, addr = sock.accept()
                        fds.append(csock)
                    elif is_reading_query_image and cur_serve_sock is not None and c == cur_serve_sock.fileno():
                        # Avoid receiving from client that is send query image
                        continue
                    else:
                        # Listen from client
                        # logging.info(f"got track_conf from {c}")

                        # Receieve current track confidence
                        cur_track_conf = recv_info(sock, 10)
                        cur_track_conf = float(cur_track_conf.decode())
                        _ = recv_info(sock, 1)
                        # logging.info(f"cur track conf ({c}) : {cur_track_conf}")

                        with min_track_conf_lock:
                            # Assign the client to server later according to track confidence
                            if min_track_conf is None or cur_track_conf < min_track_conf:
                                min_track_conf, next_serve_sock = cur_track_conf, sock
            except:
                try:
                    # Client disconnected
                    with min_track_conf_lock:
                        min_track_conf, next_serve_sock = None, None
                    logging.info(f"connection {c} ends in client_connection_thread")
                    fds.remove(sock)
                    # sock.close()
                    sock.shutdown(socket.SHUT_RDWR)
                except:
                    print('Exception during client disconnecting in client_connection_thread')
                    pass


def master_thread():
    global min_track_conf
    global next_serve_sock
    global is_reading_query_image
    global cur_serve_sock
    global num_loc
    min_track_conf, next_serve_sock, cur_serve_sock = None, None, None
    with is_reading_query_image_lock:
        is_reading_query_image = False

    # Start client connection thread
    t = threading.Thread(target=client_connection_thread)
    t.start()

    logs = defaultdict(lambda: defaultdict(list))
    logging.info("--------server is ready--------")

    while True:
        try:
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
                elif '.7_pairs-query-netvlad20' in old_item:
                    # Remove all .7_pairs-query-netvlad20 items
                    os.unlink(outputs / old_item)

            with min_track_conf_lock:
                if next_serve_sock is None:
                    continue
                else:
                    cur_serve_sock = next_serve_sock
                    sock = cur_serve_sock
                    min_track_conf = None
                    next_serve_sock = None # Delete this line for better single-client experience

            c = sock.fileno()
            logging.info(f"now serving {c} ({sock})")
            with is_reading_query_image_lock:
                is_reading_query_image = True
            
            # Ask the client to send query
            sock.send('send_query'.encode())
            
            # Make sure to wait until the client starts to send query
            recv_data = None
            while (recv_data != 'send_query'):
                recv_data = recv_info(sock, 10).decode()
                _ = recv_info(sock, 1)

            logging.info(f"Got query from {c}")

            # Read metadata
            metadata_size = recv_info(sock, 10).decode()
            _ = recv_info(sock, 1)
            logging.info(f"metadata size: {metadata_size}")

            metadata_str = recv_info(sock, int(metadata_size) - 1).decode()
            _ = recv_info(sock, 1)
            metadata = {}
            for data in metadata_str.split('|'):
                data = data.split('=')
                metadata[data[0]] = data[1]
            logging.info(f"metadata: {metadata}")

            # Parse metadata
            frame_idx = int(metadata['frame_idx'])
            focal_length = str(metadata['focal_length'])
            img_size = int(metadata['img_size'])
            cur_fused_pose = get_fused_pose_from_arpose_str(metadata['cur_fused_pose'])
            cur_track_conf = float(metadata['cur_track_conf'])

            # Set flags
            image_retrieval_by = 'global_feature'
            if cur_track_conf > 0.8 and cur_fused_pose is not None:
                image_retrieval_by = 'fused_pose'

            feature_matching_by = 'superpoint+NN'
            if cur_track_conf > 0.4:
                feature_matching_by = 'optical_flow'

            # image_retrieval_by = 'global_feature'     #
            # feature_matching_by = 'superpoint+NN'     #
            
            num_loc = 1 if feature_matching_by == 'optical_flow' else num_loc

            print(image_retrieval_by, feature_matching_by)


            # Image
            query = recv_info(sock, int(img_size))
            query = np.fromstring(query, np.uint8)
            query_img = cv2.imdecode(query, cv2.IMREAD_COLOR)
            logging.info(f"image shape: {query_img.shape}")

            with is_reading_query_image_lock:
                is_reading_query_image = False

            # Finish receiving query from client
            # Start to localize
   
            # Image pre-process
            img_name = f"q_{frame_idx}.jpg"
            img_path = Path(images, img_name)
            cv2.imwrite(str(img_path), query_img )
            # recv_img_path = Path(recv_images, img_name)
            # cv2.imwrite(str(recv_img_path), query_img)

            # Intrinsic pre-process
            query_info = arcore_parse_intrinsic(query_img.shape, focal_length)
            print('query info:', query_info)

            # Outputs pre-process
            feat_output = Path(outputs, "query_data")

            t_1 = time.monotonic()

            features = extract_features.main(feature_conf, images, feat_output)

            t_2 = time.monotonic()

            if image_retrieval_by == 'fused_pose':
                # Retrieve image by fused pose
                pairs_from_img_pose(img_name, sfm_model_poses, cur_fused_pose, loc_pairs, num_loc)
            else:
                # Retrieve image by global feature
                global_descriptors = extract_features.main(retrieval_conf, images, feat_output, model=netvlad)
                pairs_from_retrieval.main(
                    global_descriptors, loc_pairs, num_loc,
                    query_prefix='', db_prefix='', db_descriptors=db_gdesc, db_names=db_names) 

            t_4 = time.monotonic()

            if feature_matching_by == 'optical_flow':
                loc_matches = match_features_from_flow(sfm_model, loc_pairs, features, flow_features, flow_pads, scaled_db_imgs,
                                                    matches_path=Path(outputs, f"{features.stem}_{matcher_conf['output']}_{loc_pairs.stem}"),
                                                    query_img=query_img, flow_model=flow_model, args=args)
            else:
                loc_matches = match_features.main(matcher_conf, loc_pairs, features, matches=
                                                Path(outputs, f"{features.stem}_{matcher_conf['output']}_{loc_pairs.stem}"), 
                                                features_ref=db_feats, name2ref = db_local_name2ref)

            t_5 = time.monotonic()
            
            poses = localize_sfm.main(
                sfm_model,
                [(img_name, query_info)],
                loc_pairs,
                features,
                loc_matches,
                results,
                covisibility_clustering=True)  # not required with SuperPoint+SuperGlue

            # Finish localization
            # Send scale to client
            sock.send('curr_scale'.encode())
            sock.send(str('{:.19}'.format(float(scale))).zfill(20)[:20].encode())

            # Send pose back to client
            if img_name in poses:
                qvec, tvec = poses[img_name]
                result = pose_postproc(qvec, tvec)
                sock.send('result_mtx'.encode())
                sock.send(result.encode())
                sock.send(str('{:10}'.format(frame_idx)).encode())
                t_6 = time.monotonic()

                # Recording logs
                logs[c]["t_feats"].append(t_2 - t_1)   
                # logs[c]["t_global_feats"].append(t_3 - t_2)
                # logs[c]["t_retrieval"].append(t_4 - t_3)
                logs[c]["t_retrieval"].append(t_4 - t_2)
                logs[c]["t_match"].append(t_5 - t_4)
                logs[c]["t_localize"].append(t_6 - t_5)
                logs[c]["t_all"].append(t_6-t_1)

            #remove img and feature data of query 
            features.unlink()
            if image_retrieval_by == 'global_feature':
                global_descriptors.unlink()
            img_path.unlink()

            with min_track_conf_lock:
                cur_serve_sock = None

        except:
            try:
                # Client disconnected
                with is_reading_query_image_lock:
                    is_reading_query_image = False
                with min_track_conf_lock:
                    min_track_conf, next_serve_sock = None, None
                # sock.shutdown(socket.SHUT_RDWR)
                # sock.close()
                logging.info(f"connection {c} ends in master_thread")

                with open('collabloc_time.txt', 'a') as f:
                    now = datetime.now()
                    flag = datetime.strftime(now, "%y-%m-%d %H:%M:%S")
                    f.write('\n========\n' + flag + ' ' + '\n')
                    f.write(f'config: dataset={dataset}, outputs={outputs}, num_loc={num_loc}, scale={scale}, preload_flow_feats={preload_flow_feats}\n')
                    t_feats = logs[c]["t_feats"]
                    # t_global_feats = logs[c]["t_global_feats"]
                    t_retrieval = logs[c]["t_retrieval"]
                    t_match = logs[c]["t_match"]
                    t_localize = logs[c]["t_localize"]
                    t_all = logs[c]["t_all"]
                    # for log in zip(t_feats, t_global_feats, t_retrieval, t_match, t_localize):
                    for log in zip(t_feats, t_retrieval, t_match, t_localize):
                        f.write(f"feat: {log[0]:.5f} retrieval: {log[1]:.5f} match: {log[2]:.5f} localize: {log[3]:.5f}\n")
                    if len(t_feats) != 0:
                        # f.write(f"\nAVG feat : {sum(t_feats)/len(t_feats):.5f} global_feat: {sum(t_global_feats)/len(t_global_feats):.5f} retrieval: {sum(t_retrieval)/len(t_retrieval):.5f} match: {sum(t_match)/len(t_match):.5f} localize: {sum(t_localize)/len(t_localize):.5f}\n")
                        f.write(f"\nAVG feat: {sum(t_feats)/len(t_feats):.5f} retrieval: {sum(t_retrieval)/len(t_retrieval):.5f} match: {sum(t_match)/len(t_match):.5f} localize: {sum(t_localize)/len(t_localize):.5f}\n")
                        f.write(f"total AVG: {sum(t_all)/len(t_feats):.5f}\n")
            except:
                print('Exception during client disconnecting in master_thread')
                pass


def main():
    t = threading.Thread(target=master_thread)
    t.start()


if __name__ == '__main__':
    main()
