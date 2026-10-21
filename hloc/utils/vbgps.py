import logging
from pathlib import Path
from typing import Union, Optional, Dict
from .read_write_model import read_model
import numpy as np
import quaternion
import torch
import cv2
import h5py
import os
import logging
from tqdm import tqdm

from ..utils.parsers import names_to_pair, parse_retrieval

from RAFT.vbgps import get_flow, get_flow_feature_and_pad

# dshape is the same shape as query image: (480, 640, 3)
# dsize is the same size as query image: (640, 480)
# dshape = query_img.shape
# dsize = (dshape[1], dshape[0]) # (x, y)
dshape = (480, 640, 3)
dsize = (dshape[1], dshape[0]) # (x, y)


class Pose():
    def __init__(self, q, t, ptype='normal'):
        self.type = ptype
        if ptype == 'colmap':
            colmap_quat = np.quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            colmap_pos = np.array([float(t[0]), float(t[1]), float(t[2])])
            quat = colmap_quat.inverse()
            quat_arr = quaternion.as_float_array(quat)
            pos_arr = np.matmul(-quaternion.as_rotation_matrix(quat), colmap_pos)
            self.qw = quat_arr[0]
            self.qx = quat_arr[1]
            self.qy = quat_arr[2]
            self.qz = quat_arr[3]
            self.x = pos_arr[0]
            self.y = pos_arr[1]
            self.z = pos_arr[2]
        elif ptype == 'arcore':
            ar_quat = np.quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            quat = ar_quat * np.quaternion(0, -1, 0, 0)
            quat_arr = quaternion.as_float_array(quat)
            self.qw = quat_arr[0]
            self.qx = quat_arr[1]
            self.qy = quat_arr[2]
            self.qz = quat_arr[3]
            self.x = float(t[0])
            self.y = float(t[1])
            self.z = float(t[2])
        else:
            self.qw = float(q[0])
            self.qx = float(q[1])
            self.qy = float(q[2])
            self.qz = float(q[3])
            self.x = float(t[0])
            self.y = float(t[1])
            self.z = float(t[2])

    def __str__(self):
        return f'q:{self.t()}, t:{self.q()}'

    def __repr__(self):
        return self.__str__()

    def t(self):
        return np.array([self.x, self.y, self.z])

    def q(self):
        return np.quaternion(self.qw, self.qx, self.qy, self.qz)

    def rot_mtx(self):
        return quaternion.as_rotation_matrix(self.q())

    def dist_to(self, rhs):
        # Calculate distance between two positions
        return np.linalg.norm(self.t() - rhs.t())

    def angle_to(self, rhs):
        # Calculate angle between two quaternions
        # Referenced to function `rotation_error` in TrianFlow:
        # https://github.com/B1ueber2y/TrianFlow/blob/7234224ed73ac724a2cce9da41f712f5446c5ea6/core/evaluation/eval_odom.py#L123
        pose_error = quaternion.as_rotation_matrix(self.q().inverse() * rhs.q())
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        rot_error_deg = np.degrees(rot_error)
        return rot_error_deg

        # This might be wrong since it ignored rotations around z-axis:
        # v1 = np.dot(quaternion.as_rotation_matrix(q1), [0, 0, 1])
        # v2 = np.dot(quaternion.as_rotation_matrix(q2), [0, 0, 1])
        # angle = np.arccos(np.sum(np.array([e1 * e2 for e1, e2 in zip(v1, v2)])))
        # deg = np.degrees(angle)
        # return deg

    # def translation(self):
    #     return self.t()

    # def rotatation(self):
    #     return self.q()

    # def rotatation_matrix(self):
    #     return self.rot_mtx()

    def to_formatted_str(self, format_str: str) -> str:
        return format_str.replace(
            '%qw', str(self.qw)).replace('%qx', str(self.qx)).replace('%qy', str(self.qy)).replace('%qz', str(self.qz)).replace(
            '%tx', str(self.x)).replace('%ty', str(self.y)).replace('%tz', str(self.z))

    def to_arcore_pose_str(self) -> str:
        return self.to_formatted_str('{%txf, %tyf, %tzf} {%qxf, %qyf, %qzf, %qwf}')


def parse_arcore_pose_str(pose_str: str) -> list:
    for c in ['q', 't', 'x', 'y', 'z', 'w', ':', '[', ']', ',']:
        pose_str = pose_str.replace(c, '')
    return pose_str.split()


def get_fused_pose_from_arpose_str(pose_str: str) -> Union[Pose, None]:
    if pose_str == 'null':
        return None
    tok = parse_arcore_pose_str(pose_str)
    q = [tok[6], tok[3], tok[4], tok[5]]
    t = [tok[0], tok[1], tok[2]]
    return Pose(q, t)


def get_poses_from_model(reference_sfm: Union[Path, tuple], images_path) -> dict:
    if isinstance(reference_sfm, Path):
        assert reference_sfm.exists(), reference_sfm
        _, db_images, _ = read_model(str(reference_sfm))
    elif isinstance(reference_sfm, tuple):
        _, db_images, _ = reference_sfm

    existed_image_names = os.listdir(images_path)

    db_images_poses = {}
    for i, image in db_images.items():
        if image.name in existed_image_names:
            q = image.qvec
            t = image.tvec
            db_images_poses[i] = {
                'name': image.name,
                'pose': Pose(q, t, 'colmap'),
            }
    return db_images_poses


def load_flow_features_and_pads(outputs_path, args):
    feature_filename = 'feats-raft-small.npy' if args.small else 'feats-raft.npy'
    pad_filename = 'pads-raft-small.npy' if args.small else 'pads-raft.npy'
    flow_features = np.load(outputs_path / feature_filename, allow_pickle=True).item()
    flow_pads = np.load(outputs_path / pad_filename, allow_pickle=True).item()
    return flow_features, flow_pads


def pairs_from_img_pose(img_name: str, db_images_poses: dict, img_pose: Pose, output: Path, num_matched: int) -> None:
    logging.info('Extracting image pairs from pose.')

    similar_list = []
    for i, values in db_images_poses.items():
        pose = values['pose']
        dist = img_pose.dist_to(pose)
        angle = img_pose.angle_to(pose)
        similar_score = dist * 10 + angle * 1
        similar_list.append((similar_score, pose, values['name']))
    similar_list.sort()

    # Distributed retrieval
    # Select images to be retrieved
    # Note that the algorithm does not only select the top n images
    # Images with less similar_score may also be selected by a lower chance
    pairs = []
    selected_rank = 0
    for i in range(num_matched):
        selected_rank += i // max((num_matched // 4), 1) + 1
        if selected_rank >= len(similar_list):
            break
        _, _, retrieved_img_name = similar_list[selected_rank]
        pair = (img_name, retrieved_img_name)
        pairs.append(pair)

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

    # Below is only for testing
    # print('--------')
    # print(img_pose.to_formatted_str('%qw %qx %qy %qz %tx %ty %tz'))
    # for similar_score, retrieved_img_pose, retrieved_img_name in similar_list[:num_matched]:
    #     print(similar_score, retrieved_img_pose.to_formatted_str('%qw %qx %qy %qz %tx %ty %tz'))
    #     print('retr_by_fused_pose', '\n'.join(' '.join([i, j]) for i, j in pairs))
    # print('========')


def match_features_from_flow(reference_sfm, pairs_path: Path, features_path: Path, flow_features: dict, flow_pads: dict, scaled_db_imgs: dict, matches_path: Path, query_img, flow_model, args):
    assert pairs_path.exists(), pairs_path
    assert features_path.exists(), features_path

    if isinstance(reference_sfm, Path):
        assert reference_sfm.exists(), reference_sfm
        _, db_images, _ = read_model(str(reference_sfm))
    elif isinstance(reference_sfm, tuple):
        _, db_images, _ = reference_sfm
    db_name_to_id = {image.name: i for i, image in db_images.items()}

    matches_path.parent.mkdir(exist_ok=True, parents=True)
    
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]

    # Only for visualize
    # import time
    # def tmp_mark_points(img, coords: list, thickness=5, color=[255, 0, 0]):
    #     thick_minus = thickness // 2
    #     thick_plus = thickness - thick_minus
    #     for coord in coords:
    #         x, y = int(coord[0]), int(coord[1])
    #         img[y-thick_minus:y+thick_plus, x-thick_minus:x+thick_plus] = np.full(
    #             img[y-thick_minus:y+thick_plus, x-thick_minus:x+thick_plus].shape,
    #             fill_value=color,
    #             dtype=np.int
    #         )
    #     return img

    # Get query image features
    query_img_fmap, _ = get_flow_feature_and_pad(query_img, flow_model)

    for pair_idx, (name0, name1) in enumerate(pairs, 1):
        pair = names_to_pair(name0, name1)
        feature_file = h5py.File(features_path, 'r')
        query_img_kpq = np.array(feature_file[name0]['keypoints'].__array__(), dtype=int)

        # Get db image features
        if name1 in flow_features:
            scaled_db_img_pad = flow_pads[name1]['pad']
            scaled_db_img_fmap = flow_features[name1]
            db_img_shape = flow_pads[name1]['old_shape']
        else:
            scaled_db_img = scaled_db_imgs[name1]['img']
            fmap, pad_img = get_flow_feature_and_pad(scaled_db_img, flow_model)
            scaled_db_img_pad = pad_img
            scaled_db_img_fmap = fmap
            db_img_shape = scaled_db_imgs[name1]['old_shape']
        
        # Get db image keypoints
        db_img_scale = np.array([db_img_shape[1] / dsize[0], db_img_shape[0] / dsize[1]]) # (x, y)
        db_img_kpq = db_images[db_name_to_id[name1]].xys
        scaled_db_img_kpq = np.divide(db_img_kpq, db_img_scale)

        # Get optical flow
        flow = get_flow(scaled_db_img_pad, scaled_db_img_fmap, query_img_fmap, flow_model, iters=5, args=args)

        # Calculate the destinations of local features via optical flow
        scaled_db_img_kpq_flow_dst = np.full((dshape[0], dshape[1]), fill_value=-1, dtype=np.int)
        for i, kp in enumerate(scaled_db_img_kpq):
            f = flow[int(kp[1]), int(kp[0])]
            dst_x, dst_y = int(kp[0] + f[0]), int(kp[1] + f[1])

            # Expand some pixes on each x and y axis to make a larger area for matching
            # Mark this area on scaled_db_img_kpq_flow_dst
            # TODO: Fix this algorithm
            val = 2
            scaled_db_img_kpq_flow_dst[dst_y - val:dst_y + val, dst_x - val:dst_x + val] = np.full(
                scaled_db_img_kpq_flow_dst[dst_y - val:dst_y + val, dst_x - val:dst_x + val].shape,
                fill_value=i,
                dtype=np.int
            )

        # t6 = time.monotonic()
        # tmp_query_img = query_img.copy()

        matches = []
        for p in query_img_kpq:
            x, y = int(p[0]), int(p[1])
            db_img_kpq_idx = scaled_db_img_kpq_flow_dst[y, x]
            matches.append(db_img_kpq_idx)
            
            # # Only for visualize
            # if db_img_kpq_idx != -1:
            #     tmp_dbp = scaled_db_img_kpq[db_img_kpq_idx]
            #     tmp_dbx, tmp_dby = int(tmp_dbp[0]), int(tmp_dbp[1])
            #     tmp_query_img = cv2.line(tmp_query_img, (x, y), (tmp_dbx, tmp_dby), (255, 255, 0), 1)
            #     tmp_query_img = tmp_mark_points(tmp_query_img, [(x, y)], color=[0, 255, 0], thickness=3)
            # else:
            #     # tmp_query_img = tmp_mark_points(tmp_query_img, [(x, y)], color=[255, 0, 0], thickness=3)
            #     pass

        matches = np.array(matches, dtype=np.int16)
        
        # t7 = time.monotonic()
        
        with h5py.File(str(matches_path), 'a') as fd:
            grp = fd.create_group(pair)
            grp.create_dataset('matches0', data=matches)

        # Only for visualize
        # for p in tmp_scaled_db_img_kpq_flow_dst:
        #     tmp_query_img = tmp_mark_points(tmp_query_img, [p], color=[255, 255, 255], thickness=1)

        # Only for testing
        # out_img = np.concatenate([tmp_flow_visual, db_img, query_img], axis=1) 
        # cv2.imwrite(f'{args.out}/vis_{pair_idx}_{name0}_{name1}', scaled_db_img[:, :, [2,1,0]] * 0.3 + tmp_query_img[:, :, [2,1,0]] * 0.7)
        # cv2.imwrite(f'{args.out}/vis_{pair_idx}_{name0}_{name1}', tmp_query_img[:, :, [2,1,0]])
        # cv2.imwrite(f'{args.out}/db_img_{pairs[0][1]}', scaled_db_img[:, :, [2,1,0]])
        # cv2.imwrite(f'{args.out}/query_img_{pairs[0][1]}', query_img[:, :, [2,1,0]])
        # cv2.imwrite(f'{args.out}/flow_visual_{pairs[0][1]}', tmp_flow_visual[:, :, [2,1,0]])
        # print('match_features_from_flow', matches, len(matches))

        # t8 = time.monotonic()

        # print()
        # print(1, round(t2 - t1, 3))
        # print(2, round(t3 - t2, 3))
        # print(3, round(t4 - t3, 3))
        # print(4, round(t5 - t4, 3))
        # print(5, round(t6 - t5, 3))
        # print(6, round(t7 - t6, 3))
        # print(7, round(t8 - t7, 3))

    return matches_path
