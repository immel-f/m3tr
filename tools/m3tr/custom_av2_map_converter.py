from lanelet2.ml_converter import MapDataInterface, LineStringType, toPointMatrix
from lanelet2.core import (BasicPoint3d, Lanelet, LaneletMap,
                           LineString3d, Point2d, Point3d, getId)
import lanelet2
from collections import defaultdict
from copy import deepcopy
import signal
from functools import partial
from multiprocessing import Pool
import multiprocessing
from random import sample
import random
import time
import mmcv
import logging
from pathlib import Path
from os import path as osp
import os
import av2.geometry.utm
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import ArgoverseStaticMap
from tqdm import tqdm
import time
import argparse
import networkx as nx
from av2.map.map_primitives import Polyline
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from shapely.strtree import STRtree
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from av2.geometry.se3 import SE3
import numpy as np
import math
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance, KDTree
import warnings
warnings.filterwarnings("ignore")


try:
    from tools.m3tr.map_element_utils import calc_masked_elements
except:
    from map_element_utils import calc_masked_elements


CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
             'ring_rear_right', 'ring_rear_left', 'ring_side_right', 'ring_side_left',
             # 'stereo_front_left', 'stereo_front_right',
             ]
# some fail logs as stated in av2
# https://github.com/argoverse/av2-api/blob/05b7b661b7373adb5115cf13378d344d2ee43906/src/av2/map/README.md#training-online-map-inference-models
FAIL_LOGS = [
    # official
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
    # observed
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    # observed ll2_custom
    '8940f5f1-13e0-3094-99ba-da2d17639774',
    'c08279c0-10b4-3d21-b13f-a1c1a0b87f8b',
    'c96a09c8-46ed-391f-8a66-c46fa8b76029'
]

AV2_LANEMARKTYPE_TO_LL2 = {
    LaneMarkType.DASH_SOLID_YELLOW: 'dashed_solid',
    LaneMarkType.DASH_SOLID_WHITE: 'dashed_solid',

    LaneMarkType.DASHED_WHITE: 'dashed',
    LaneMarkType.DASHED_YELLOW: 'dashed',

    LaneMarkType.DOUBLE_SOLID_YELLOW: 'solid',
    LaneMarkType.DOUBLE_SOLID_WHITE: 'solid',

    LaneMarkType.DOUBLE_DASH_YELLOW: 'dashed',
    LaneMarkType.DOUBLE_DASH_WHITE: 'dashed',

    LaneMarkType.SOLID_YELLOW: 'solid',
    LaneMarkType.SOLID_WHITE: 'solid',

    LaneMarkType.SOLID_DASH_WHITE: 'solid_dashed',
    LaneMarkType.SOLID_DASH_YELLOW: 'solid_dashed',

    LaneMarkType.SOLID_BLUE: 'solid',

    LaneMarkType.NONE: 'virtual',

    LaneMarkType.UNKNOWN: 'unknown'
}

RANDOM_MASKING_LIST = [
    ['ego_lane'],
    ['ego_road'],
    ['centerline'],
    ['ped_crossing'],
    ['boundary'],
    ['divider_solid', 'divider_dashed'],
    ['boundary', 'divider_dashed', 'divider_solid', 'ped_crossing'],
    ['centerline', 'divider_dashed', 'divider_solid', 'ped_crossing'],
    ['boundary', 'centerline', 'divider_dashed', 'divider_solid', 'ped_crossing'],
]


def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-root',
        type=str,
        help='specify the output path of the generated annotations')
    parser.add_argument(
        '--pc-range',
        type=float,
        nargs='+',
        default=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
        help='specify the perception point cloud range')
    parser.add_argument(
        '--nproc',
        type=int,
        default=64,
        required=False,
        help='workers to process data')
    parser.add_argument(
        '--use-mixed',
        type=bool,
        default=False,
        required=False,
        help='Use the mixed divider type (solid dashed or dashed solid) for labels. If false (default), mixed dividers will be classified as solid')
    parser.add_argument(
        '--use-virtual',
        type=bool,
        default=False,
        required=False,
        help='Use the virtual divider type for labels. If false, virtual dividers will be excluded from labels')
    parser.add_argument(
        '--masked-elements',
        nargs='+',
        type=str,
        default=None,
        required=False,
        help="""
             Elements that should be masked from the map information given to the network. 
             If none are given, the network will not recieve any map information during training.
             Possible Options:
             ego_lane: masks out all labels associated with the ego lane
             ego_road: masks out all labels associated with the ego road
             random: randomly selects a masking type for a sample
             random_whole_dataset: duplicates each sample for each available masking type (e.g. 8 masking types = 
             8x the dataset annotations stacked, once for each masking type)
             <list of element types>: masks out all elements with the specified type, e.g. divider_solid, divider_dashed, centerline etc.
             """)
    args = parser.parse_args()
    return args

# def track_job(job, update_interval=2):
#     while job._number_left > 0:
#         print("Tasks remaining = {0}".format(
#         job._number_left * job._chunksize))
#         time.sleep(update_interval)


def timeout_handler(timeout_msg, signum, frame):
    print(timeout_msg)
    raise Exception("end of time")


def create_av2_infos_mp(root_path,
                        info_prefix,
                        dest_path=None,
                        split='train',
                        num_multithread=96,
                        pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                        use_mixed=False,
                        use_virtual=True,
                        masked_elements=None):
    """Create info file of av2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dest_path (str): Path to store generated file, default to root_path
        split (str): Split of the data.
            Default: 'train'
    """
    root_path = osp.join(root_path, split)
    if dest_path is None:
        dest_path = root_path

    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
    log_ids = list(loader.get_log_ids())
    # import pdb;pdb.set_trace()
    for l in FAIL_LOGS:
        if l in log_ids:
            log_ids.remove(l)

    print('collecting samples...')
    start_time = time.time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')
    print("iterations needed in total: " + str(len(log_ids)))

    # to supress logging from av2.utils.synchronization_database
    sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    prev_level = sdb_logger.level
    sdb_logger.setLevel(logging.CRITICAL)

    # # FIXME: need to check the order
    # pool = Pool(num_multithread)
    # fn = partial(get_data_from_logid, loader=loader, data_root=root_path, pc_range=pc_range)
    # rt = pool.map_async(fn, log_ids)
    # # track_job(rt)
    # pool.close()
    # pool.join()
    # results = rt.get()

    log_ids_track = deepcopy(log_ids)

    results = []
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path, pc_range=pc_range,
                 use_mixed=use_mixed, use_virtual=use_virtual, masked_elements=masked_elements)
    pool = multiprocessing.Pool(num_multithread)
    try:
        for samples, discarded, log_id in tqdm(pool.imap(fn, log_ids)):
            results.append((samples, discarded))
            log_ids_track.remove(log_id)
            if len(log_ids_track) < 5:
                print(log_ids_track)
        # for log_id in tqdm(log_ids):
        #     samples, discarded = fn(log_id)
        #     results.append((samples, discarded))
        #     log_ids_track.remove(log_id)
        #     if len(log_ids_track) < 5:
        #         print(log_ids_track)
    except KeyboardInterrupt:
        logging.warning("got Ctrl+C")
    finally:
        pool.terminate()
        pool.join()

    # fn = partial(get_data_from_logid, loader=loader, data_root=root_path, pc_range=pc_range)
    # results = [fn(log_id) for log_id in tqdm(log_ids[0:1])]

    samples = []
    discarded = 0
    sample_idx = 0
    for _samples, _discarded in results:
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        samples += _samples
        discarded += _discarded

    sdb_logger.setLevel(prev_level)
    print(f'{len(samples)} available samples, {discarded} samples discarded')

    print('collected in {}s'.format(time.time()-start_time))
    infos = dict(samples=samples)

    info_path = osp.join(dest_path,
                         '{}_map_infos_{}.pkl'.format(info_prefix, split))
    print(f'saving results to {info_path}')
    mmcv.dump(infos, info_path)
    # mmcv.dump(samples, info_path)


def get_divider(avm):
    divider_list = []
    for ls in avm.get_scenario_lane_segments():
        for bound_type, bound_city in zip([ls.left_mark_type, ls.right_mark_type], [ls.left_lane_boundary, ls.right_lane_boundary]):
            if bound_type not in [LaneMarkType.NONE,]:
                divider_list.append(bound_city.xyz)
    return divider_list


def get_boundary(avm):
    boundary_list = []
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)
    return boundary_list


def get_ped(avm):
    ped_list = []
    for pc in avm.get_scenario_ped_crossings():
        ped_list.append(pc.polygon)
    return ped_list


def get_data_from_logid(log_id,
                        loader: AV2SensorDataLoader,
                        data_root,
                        pc_range=[-30.0, -15.0, -5.0, 30.0, 15.0, 3.0],
                        use_mixed=False,
                        use_virtual=True,
                        masked_elements=None):
    samples = []
    discarded = 0

    timeout_msg = "TIMEOUT ON get_data_from_logid WITH LOG_ID: " + str(log_id)
    signal.signal(signal.SIGALRM, partial(timeout_handler, timeout_msg))
    signal.alarm(700)

    log_map_dirpath = Path(osp.join(data_root, log_id, "map"))
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(
            f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    vector_data_json_path = vector_data_fname
    avm = ArgoverseStaticMap.from_json(vector_data_json_path)
    # We use lidar timestamps to query all sensors.
    # The frequency is 10Hz
    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]

    for ts in cam_timestamps:
        cam_ring_fpath = [loader.get_closest_img_fpath(
            log_id, cam_name, ts
        ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cam_timestamp_ns = int(cam_ring_fpath[i].stem)
            cam_city_SE3_ego = loader.get_city_SE3_ego(
                log_id, cam_timestamp_ns)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
                e2g_translation=cam_city_SE3_ego.translation,
                e2g_rotation=cam_city_SE3_ego.rotation,
            )

        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        info = dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams,
            lidar_path=str(lidar_fpath),
            # map_fpath=map_fname,
            timestamp=str(ts),
            log_id=log_id,
            token=str(log_id+'_'+str(ts)))

        if 'random_whole_dataset' in masked_elements:
            for masked_elements_random in RANDOM_MASKING_LIST:
                map_anno = extract_local_map(
                    avm, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements_random)
                info_cpy = deepcopy(info)
                info_cpy["annotation"] = map_anno
                samples.append(info_cpy)
        elif 'random' in masked_elements:
            masked_elements_random = random.choice(RANDOM_MASKING_LIST)
            map_anno = extract_local_map(
                avm, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements_random)
            info["annotation"] = map_anno
            samples.append(info)
        else:
            map_anno = extract_local_map(
                avm, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements)
            info["annotation"] = map_anno
            samples.append(info)

    signal.alarm(0)

    return samples, discarded, log_id


def intersecting_boundaries(divider, boundaries, tolerance=0.5):
    bd_indices = []
    for i, bd in enumerate(boundaries):
        if LineString(divider).distance(LineString(bd)) < tolerance:
            bd_indices.append(i)

    return bd_indices


def has_min_length(divider, min_length=1.0):
    if LineString(divider).length < min_length:
        return False
    else:
        return True


def extract_local_map(avm, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, masked_elements):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_size = (patch_h, patch_w)
    map_pose = e2g_translation[:2]
    rotation = Quaternion._from_matrix(e2g_rotation)
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    city_SE2_ego = SE3(e2g_rotation, e2g_translation)
    ego_SE3_city = city_SE2_ego.inverse()

    result_dict = generate_nearby_dividers_and_centerlines(
        avm, patch_box, patch_angle, ego_SE3_city, use_mixed, masked_elements)

    map_anno = dict(
        divider_dashed=[],
        divider_solid=[],
        divider_mixed=[],
        divider_virtual=[],
        ped_crossing=[],
        boundary=[],
        centerline=[],
        divider_dashed_map_data_idx=[],
        divider_solid_map_data_idx=[],
        divider_mixed_map_data_idx=[],
        divider_virtual_map_data_idx=[],
        centerline_map_data_idx=[],
        divider_dashed_masked=[],
        divider_solid_masked=[],
        divider_mixed_masked=[],
        divider_virtual_masked=[],
        centerline_masked=[],
        boundary_masked=[]
    )

    map_anno['ped_crossing'] = extract_local_ped_crossing(
        avm, ego_SE3_city, patch_box, patch_angle, patch_size)
    map_anno['boundary'] = extract_local_boundary(
        avm, ego_SE3_city, patch_box, patch_angle, patch_size)
    map_anno['centerline'] = result_dict['centerline']
    map_anno['divider_dashed'] = result_dict['divider_dashed']
    map_anno['divider_solid'] = result_dict['divider_solid']
    map_anno['divider_dashed_map_data_idx'] = result_dict['divider_dashed_map_data_idx']
    map_anno['divider_solid_map_data_idx'] = result_dict['divider_solid_map_data_idx']
    map_anno['centerline_map_data_idx'] = result_dict['centerline_map_data_idx']
    map_anno['divider_dashed_masked'] = result_dict['divider_dashed_masked']
    map_anno['divider_solid_masked'] = result_dict['divider_solid_masked']
    map_anno['centerline_masked'] = result_dict['centerline_masked']

    boundary_masked = np.array([False for bd in map_anno['boundary']])
    divider_dashed_intersecting_bds = [intersecting_boundaries(div, map_anno['boundary']) for i, div in enumerate(
        result_dict['divider_dashed']) if len(intersecting_boundaries(div, map_anno['boundary'])) > 0 and result_dict['divider_dashed_masked'][i]]
    divider_dashed_intersecting_bds = np.array(
        [idx for indices in divider_dashed_intersecting_bds for idx in indices], dtype=int).flatten()
    divider_solid_intersecting_bds = [intersecting_boundaries(div, map_anno['boundary']) for i, div in enumerate(
        result_dict['divider_solid']) if len(intersecting_boundaries(div, map_anno['boundary'])) > 0 and result_dict['divider_solid_masked'][i]]
    divider_solid_intersecting_bds = np.array(
        [idx for indices in divider_solid_intersecting_bds for idx in indices], dtype=int).flatten()

    boundary_masked[divider_dashed_intersecting_bds] = True
    boundary_masked[divider_solid_intersecting_bds] = True

    divider_dashed_mask = np.array([True if len(intersecting_boundaries(div, map_anno['boundary'])) == 0 and has_min_length(
        div) else False for div in map_anno['divider_dashed']], dtype=bool)
    map_anno['divider_dashed'] = [div for i, div in enumerate(
        map_anno['divider_dashed']) if divider_dashed_mask[i]]
    map_anno['divider_dashed_map_data_idx'] = map_anno['divider_dashed_map_data_idx'][divider_dashed_mask]
    map_anno['divider_dashed_masked'] = map_anno['divider_dashed_masked'][divider_dashed_mask]

    divider_solid_mask = np.array([True if len(intersecting_boundaries(div, map_anno['boundary'])) == 0 and has_min_length(
        div) else False for div in map_anno['divider_solid']], dtype=bool)
    map_anno['divider_solid'] = [div for i, div in enumerate(
        map_anno['divider_solid']) if divider_solid_mask[i]]
    map_anno['divider_solid_map_data_idx'] = map_anno['divider_solid_map_data_idx'][divider_solid_mask]
    map_anno['divider_solid_masked'] = map_anno['divider_solid_masked'][divider_solid_mask]

    if use_mixed:
        map_anno['divider_mixed'] = result_dict['divider_mixed']
        map_anno['divider_mixed_map_data_idx'] = result_dict['divider_mixed_map_data_idx']
        map_anno['divider_mixed_masked'] = result_dict['divider_mixed_masked']
        divider_mixed_mask = np.array([True if len(intersecting_boundaries(div, map_anno['boundary'])) == 0 and has_min_length(
            div) else False for div in map_anno['divider_mixed']], dtype=bool)
        map_anno['divider_mixed'] = [div for i, div in enumerate(
            map_anno['divider_mixed']) if divider_mixed_mask[i]]
        map_anno['divider_mixed_map_data_idx'] = map_anno['divider_mixed_map_data_idx'][divider_mixed_mask]
        map_anno['divider_mixed_masked'] = map_anno['divider_mixed_masked'][divider_mixed_mask]

        divider_mixed_intersecting_bds = [intersecting_boundaries(div, map_anno['boundary']) for i, div in enumerate(
            result_dict['divider_mixed']) if len(intersecting_boundaries(div, map_anno['boundary'])) > 0 and result_dict['divider_mixed_masked'][i]]
        divider_mixed_intersecting_bds = np.array(
            [idx for indices in divider_mixed_intersecting_bds for idx in indices], dtype=int).flatten()
        boundary_masked[divider_mixed_intersecting_bds] = True

    if use_virtual:
        map_anno['divider_virtual'] = result_dict['divider_virtual']
        map_anno['divider_virtual_map_data_idx'] = result_dict['divider_virtual_map_data_idx']
        map_anno['divider_virtual_masked'] = result_dict['divider_virtual_masked']
        divider_virtual_mask = np.array([True if len(intersecting_boundaries(div, map_anno['boundary'])) == 0 and has_min_length(
            div) else False for div in map_anno['divider_virtual']], dtype=bool)
        map_anno['divider_virtual'] = [div for i, div in enumerate(
            map_anno['divider_virtual']) if divider_virtual_mask[i]]
        map_anno['divider_virtual_map_data_idx'] = map_anno['divider_virtual_map_data_idx'][divider_virtual_mask]
        map_anno['divider_virtual_masked'] = map_anno['divider_virtual_masked'][divider_virtual_mask]

        divider_virtual_intersecting_bds = [intersecting_boundaries(div, map_anno['boundary']) for i, div in enumerate(
            result_dict['divider_virtual']) if len(intersecting_boundaries(div, map_anno['boundary'])) > 0 and result_dict['divider_virtual_masked'][i]]
        divider_virtual_intersecting_bds = np.array(
            [idx for indices in divider_virtual_intersecting_bds for idx in indices], dtype=int).flatten()
        boundary_masked[divider_virtual_intersecting_bds] = True

    if 'ego_lane' in masked_elements or 'ego_road' in masked_elements:
        map_anno['boundary_masked'] = boundary_masked

        ped_crossing_masked = np.array(
            [False for bd in map_anno['ped_crossing']])

        if len(map_anno['ped_crossing']) > 0 and all([len(el) for el in map_anno['ped_crossing']]):

            divider_dashed_intersecting_ped_crossing = [intersecting_boundaries(div, map_anno['ped_crossing']) for i, div in enumerate(
                result_dict['divider_dashed']) if len(intersecting_boundaries(div, map_anno['ped_crossing'])) > 0 and result_dict['divider_dashed_masked'][i]]
            divider_dashed_intersecting_ped_crossing = np.array(
                [idx for indices in divider_dashed_intersecting_ped_crossing for idx in indices], dtype=int).flatten()

            divider_solid_intersecting_ped_crossing = [intersecting_boundaries(div, map_anno['ped_crossing']) for i, div in enumerate(
                result_dict['divider_solid']) if len(intersecting_boundaries(div, map_anno['ped_crossing'])) > 0 and result_dict['divider_solid_masked'][i]]
            divider_solid_intersecting_ped_crossing = np.array(
                [idx for indices in divider_solid_intersecting_ped_crossing for idx in indices], dtype=int).flatten()

            centerline_intersecting_ped_crossing = [intersecting_boundaries(div, map_anno['ped_crossing']) for i, div in enumerate(
                result_dict['centerline']) if len(intersecting_boundaries(div, map_anno['ped_crossing'])) > 0 and result_dict['centerline_masked'][i]]
            centerline_intersecting_ped_crossing = np.array(
                [idx for indices in centerline_intersecting_ped_crossing for idx in indices], dtype=int).flatten()

            boundary_intersecting_ped_crossing = [intersecting_boundaries(div, map_anno['ped_crossing']) for i, div in enumerate(
                map_anno['boundary']) if len(intersecting_boundaries(div, map_anno['ped_crossing'])) > 0 and map_anno['boundary_masked'][i]]
            boundary_intersecting_ped_crossing = np.array(
                [idx for indices in boundary_intersecting_ped_crossing for idx in indices], dtype=int).flatten()

            ped_crossing_masked[divider_dashed_intersecting_ped_crossing] = True
            ped_crossing_masked[divider_solid_intersecting_ped_crossing] = True
            ped_crossing_masked[centerline_intersecting_ped_crossing] = True
            ped_crossing_masked[boundary_intersecting_ped_crossing] = True

        map_anno['ped_crossing_masked'] = ped_crossing_masked

    else:
        map_anno['boundary_masked'] = np.array(
            [False for bd in map_anno['boundary']])

    for el_type in masked_elements:
        if el_type == 'boundary' or el_type == 'ped_crossing':
            map_anno[el_type +
                     '_masked'] = np.array([True for bd in map_anno[el_type]])

    return map_anno


def generate_nearby_dividers_and_centerlines(avm, patch_box, patch_angle, ego_SE3_city, use_mixed, masked_elements):

    map_pose = (patch_box[0], patch_box[1], (patch_angle / 180.0) * np.pi)
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    scene_ls_list = avm.get_scenario_lane_segments()
    scene_ls_dict = dict()
    for ls in scene_ls_list:
        scene_ls_dict[ls.id] = dict(
            ls=ls,
            polygon=Polygon(ls.polygon_boundary),
            predecessors=ls.predecessors,
            successors=ls.successors
        )

    nearby_ls_dict = dict()
    for key, value in scene_ls_dict.items():
        polygon = value['polygon']
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                nearby_ls_dict[key] = value['ls']

        # nearby_ls_dict[key] = value['ls']

    ll2_map = LaneletMap()
    for key, value in nearby_ls_dict.items():
        left_line_geom = LineString(value.left_lane_boundary.xyz)
        left_line_pts = np.array(left_line_geom.coords).round(3)
        left_line_ll2_pts = []

        for pt in left_line_pts:
            left_line_ll2_pts.append(Point3d(getId(), pt[0], pt[1], pt[2]))

        left_line_ll2 = LineString3d(getId(), left_line_ll2_pts)
        left_ll2_type = AV2_LANEMARKTYPE_TO_LL2[value.left_mark_type]
        if left_ll2_type == 'virtual':
            left_line_ll2.attributes['type'] = left_ll2_type
        else:
            left_line_ll2.attributes['type'] = 'line_thin'
            left_line_ll2.attributes['subtype'] = left_ll2_type

        right_line_geom = LineString(value.right_lane_boundary.xyz)
        right_line_pts = np.array(right_line_geom.coords).round(3)
        right_line_ll2_pts = []

        for pt in right_line_pts:
            right_line_ll2_pts.append(Point3d(getId(), pt[0], pt[1], pt[2]))

        right_line_ll2 = LineString3d(getId(), right_line_ll2_pts)
        right_ll2_type = AV2_LANEMARKTYPE_TO_LL2[value.right_mark_type]
        if right_ll2_type == 'virtual':
            right_line_ll2.attributes['type'] = right_ll2_type
        else:
            right_line_ll2.attributes['type'] = 'line_thin'
            right_line_ll2.attributes['subtype'] = right_ll2_type

        lanelet = Lanelet(key, left_line_ll2, right_line_ll2)
        lanelet.attributes['subtype'] = 'road'

        ll2_map.add(lanelet)

    point_list = [(np.array([pt.x, pt.y, pt.z]), pt.id)
                  for pt in ll2_map.pointLayer]
    pts = np.row_stack([pt_tp[0] for pt_tp in point_list])
    kdtree = KDTree(pts)
    merge_radius = 0.05
    double_pt_indices = kdtree.query_ball_point(pts, merge_radius)

    for inds in double_pt_indices:
        if len(inds) < 2:
            continue

        keep_id = None
        for idx in inds:
            if keep_id is None:
                keep_id = point_list[idx][1]
            else:
                pt_id = point_list[idx][1]
                linestrings_with_pt = ll2_map.lineStringLayer.findUsages(
                    ll2_map.pointLayer[pt_id])
                for ls in linestrings_with_pt:
                    for ls_idx in range(0, len(ls)):
                        if ls[ls_idx].id == pt_id:
                            ls[ls_idx] = ll2_map.pointLayer[keep_id]

    ls_ids_list = [([pt.id for pt in ls], ls.id)
                   for ls in ll2_map.lineStringLayer]
    ls_replaced = {}
    for ls in ls_ids_list:
        for ls_comp in ls_ids_list:

            if ls_comp[1] == ls[1] or len(ls_comp[0]) != len(ls[0]) or ls_comp[1] in ls_replaced or ls[1] in ls_replaced:
                continue

            if ls[0] == ls_comp[0]:
                lanelets_with_ls = ll2_map.laneletLayer.findUsages(
                    ll2_map.lineStringLayer[ls_comp[1]])
                for ll in lanelets_with_ls:
                    if ll.leftBound.id == ls_comp[1]:
                        ll.leftBound = ll2_map.lineStringLayer[ls[1]]
                        ls_replaced[ls_comp[1]] = True
                    if ll.rightBound.id == ls_comp[1]:

                        ll.rightBound = ll2_map.lineStringLayer[ls[1]]
                        ls_replaced[ls_comp[1]] = True

            elif ls[0] == list(reversed(ls_comp[0])):
                lanelets_with_ls = ll2_map.laneletLayer.findUsages(
                    ll2_map.lineStringLayer[ls_comp[1]])
                for ll in lanelets_with_ls:
                    if ll.leftBound.id == ls_comp[1]:
                        ll.leftBound = ll2_map.lineStringLayer[ls[1]].invert()
                        ls_replaced[ls_comp[1]] = True
                    if ll.rightBound.id == ls_comp[1]:
                        ll.rightBound = ll2_map.lineStringLayer[ls[1]].invert()
                        ls_replaced[ls_comp[1]] = True

    ll2_map = lanelet2.core.createMapFromLanelets(
        [ll for ll in ll2_map.laneletLayer])

    # av2_city_name = avm.log_id[-14:-11]
    # map_origin = av2.geometry.utm.CITY_ORIGIN_LATLONG_DICT[av2.geometry.utm.CityName(av2_city_name)]
    # projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(map_origin[0], map_origin[1]))
    # lanelet2.io.write("/workspace/MapTR/gen_labels/debug_ll2_maps/" + avm.log_id + ".osm", ll2_map, projector)
    # print("Saved " + avm.log_id + ".osm")

    pos = BasicPoint3d(patch_box[0], patch_box[1], 0)
    yaw = (patch_angle / 180.0) * np.pi
    config = MapDataInterface.Configuration()
    config.submapExtentLongitudinal = patch_box[3] / 2.0
    config.submapExtentLateral = patch_box[2] / 2.0
    mDataIf = MapDataInterface(ll2_map, config)
    mDataIf.setCurrPosAndExtractSubmap(pos, yaw)
    lData = mDataIf.laneData(True)

    result_dict = {
        'divider_dashed': [],
        'divider_solid': [],
        'divider_mixed': [],
        'divider_virtual': [],
        'centerline': [],
        'divider_dashed_map_data_idx': [],
        'divider_solid_map_data_idx': [],
        'divider_mixed_map_data_idx': [],
        'divider_virtual_map_data_idx': [],
        'centerline_map_data_idx': [],
        'divider_dashed_masked_indices': [],
        'divider_solid_masked_indices': [],
        'divider_mixed_masked_indices': [],
        'divider_virtual_masked_indices': [],
        'centerline_masked_indices': [],
        'divider_dashed_masked': [],
        'divider_solid_masked': [],
        'divider_mixed_masked': [],
        'divider_virtual_masked': [],
        'centerline_masked': []
    }

    for i, div in enumerate(lData.compoundLaneDividers):
        if div.type == LineStringType.Dashed:
            for lstring in div.cutInstance:
                result_dict['divider_dashed'].append(np.array(
                    proc_line(LineString(toPointMatrix(lstring, False)),  ego_SE3_city).coords))
                result_dict['divider_dashed_map_data_idx'].append(i)
        if div.type == LineStringType.Solid:
            for lstring in div.cutInstance:
                result_dict['divider_solid'].append(np.array(
                    proc_line(LineString(toPointMatrix(lstring, False)),  ego_SE3_city).coords))
                result_dict['divider_solid_map_data_idx'].append(i)
        if div.type == LineStringType.Mixed:
            for lstring in div.cutInstance:
                if use_mixed:
                    result_dict['divider_mixed'].append(np.array(
                        proc_line(LineString(toPointMatrix(lstring, False)),  ego_SE3_city).coords))
                    result_dict['divider_mixed_map_data_idx'].append(i)
                else:
                    result_dict['divider_solid'].append(np.array(
                        proc_line(LineString(toPointMatrix(lstring, False)),  ego_SE3_city).coords))
                    result_dict['divider_solid_map_data_idx'].append(i)
        if div.type == LineStringType.Virtual:
            for lstring in div.cutInstance:
                result_dict['divider_virtual'].append(np.array(
                    proc_line(LineString(toPointMatrix(lstring, False)),  ego_SE3_city).coords))
                result_dict['divider_virtual_map_data_idx'].append(i)

    for i, cline in enumerate(lData.compoundCenterlines):
        for lstring in cline.cutInstance:
            result_dict['centerline'].append(np.array(
                proc_line(LineString(toPointMatrix(lstring, False)),  ego_SE3_city).coords))
            result_dict['centerline_map_data_idx'].append(i)

    result_dict['divider_dashed_map_data_idx'] = np.array(
        result_dict['divider_dashed_map_data_idx'])
    result_dict['divider_solid_map_data_idx'] = np.array(
        result_dict['divider_solid_map_data_idx'])
    result_dict['divider_mixed_map_data_idx'] = np.array(
        result_dict['divider_mixed_map_data_idx'])
    result_dict['divider_virtual_map_data_idx'] = np.array(
        result_dict['divider_virtual_map_data_idx'])
    result_dict['centerline_map_data_idx'] = np.array(
        result_dict['centerline_map_data_idx'])

    if masked_elements is not None:
        return calc_masked_elements(ll2_map, None, lData, result_dict, map_pose, masked_elements, include_boundaries=False)
    else:
        return result_dict


def proc_polygon(polygon, ego_SE3_city):
    # import pdb;pdb.set_trace()
    interiors = []
    exterior_cityframe = np.array(list(polygon.exterior.coords))
    exterior_egoframe = ego_SE3_city.transform_point_cloud(exterior_cityframe)
    for inter in polygon.interiors:
        inter_cityframe = np.array(list(inter.coords))
        inter_egoframe = ego_SE3_city.transform_point_cloud(inter_cityframe)
        interiors.append(inter_egoframe[:, :3])

    new_polygon = Polygon(exterior_egoframe[:, :3], interiors)
    return new_polygon


def proc_line(line, ego_SE3_city):
    # import pdb;pdb.set_trace()
    new_line_pts_cityframe = np.array(list(line.coords))
    new_line_pts_egoframe = ego_SE3_city.transform_point_cloud(
        new_line_pts_cityframe)
    line = LineString(new_line_pts_egoframe[:, :3])  # TODO
    return line


def extract_local_boundary(avm, ego_SE3_city, patch_box, patch_angle, patch_size):
    boundary_list = []
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    for da in avm.get_scenario_vector_drivable_areas():
        boundary_list.append(da.xyz)

    polygon_list = []
    for da in boundary_list:
        exterior_coords = da
        interiors = []
    #     polygon = Polygon(exterior_coords, interiors)
        polygon = Polygon(exterior_coords, interiors)
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                if new_polygon.geom_type is 'Polygon':
                    if not new_polygon.is_valid:
                        continue
                    new_polygon = proc_polygon(new_polygon, ego_SE3_city)
                    if not new_polygon.is_valid:
                        continue
                elif new_polygon.geom_type is 'MultiPolygon':
                    polygons = []
                    for single_polygon in new_polygon.geoms:
                        if not single_polygon.is_valid or single_polygon.is_empty:
                            continue
                        new_single_polygon = proc_polygon(
                            single_polygon, ego_SE3_city)
                        if not new_single_polygon.is_valid:
                            continue
                        polygons.append(new_single_polygon)
                    if len(polygons) == 0:
                        continue
                    new_polygon = MultiPolygon(polygons)
                    if not new_polygon.is_valid:
                        continue
                else:
                    raise ValueError(
                        '{} is not valid'.format(new_polygon.geom_type))

                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)

    union_segments = ops.unary_union(polygon_list)
    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    exteriors = []
    interiors = []
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments.geoms:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)

    boundary_lines = []
    for line in results:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    boundary_lines.append(np.array(single_line.coords))
            elif line.geom_type == 'LineString':
                boundary_lines.append(np.array(line.coords))
            else:
                raise NotImplementedError
    return boundary_lines


def extract_local_ped_crossing(avm, ego_SE3_city, patch_box, patch_angle, patch_size):
    ped_list = []
    for pc in avm.get_scenario_ped_crossings():
        ped_list.append(pc.polygon)

    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

    polygon_list = []
    for pc in ped_list:
        exterior_coords = pc
        interiors = []
        polygon = Polygon(exterior_coords, interiors)
        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                if new_polygon.geom_type is 'Polygon':
                    if not new_polygon.is_valid:
                        continue
                    new_polygon = proc_polygon(new_polygon, ego_SE3_city)
                    if not new_polygon.is_valid:
                        continue
                elif new_polygon.geom_type is 'MultiPolygon':
                    polygons = []
                    for single_polygon in new_polygon.geoms:
                        if not single_polygon.is_valid or single_polygon.is_empty:
                            continue
                        new_single_polygon = proc_polygon(
                            single_polygon, ego_SE3_city)
                        if not new_single_polygon.is_valid:
                            continue
                        polygons.append(new_single_polygon)
                    if len(polygons) == 0:
                        continue
                    new_polygon = MultiPolygon(polygons)
                    if not new_polygon.is_valid:
                        continue
                else:
                    raise ValueError(
                        '{} is not valid'.format(new_polygon.geom_type))

                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append(new_polygon)

    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle  # polygon as rotated rect
        rect_v_p = np.array(rect.exterior.coords)[:3]  # vector point
        rect_v = rect_v_p[1:]-rect_v_p[:-1]  # vector
        v_len = np.linalg.norm(rect_v, axis=-1)  # vector length
        longest_v_i = v_len.argmax()

        return rect_v[longest_v_i], v_len[longest_v_i]

    ped_geoms = polygon_list
    tree = STRtree(ped_geoms)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))
    final_pgeom = []
    remain_idx = [i for i in range(len(ped_geoms))]
    for i, pgeom in enumerate(ped_geoms):
        if i not in remain_idx:
            continue
        remain_idx.pop(remain_idx.index(i))
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)
        for o in tree.query(pgeom):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue
            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
            if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                final_pgeom[-1] =\
                    final_pgeom[-1].union(o)  # union parallel ped?
                # update
                remain_idx.pop(remain_idx.index(o_idx))
    for i in range(len(final_pgeom)):
        if final_pgeom[i].geom_type != 'MultiPolygon':
            final_pgeom[i] = MultiPolygon([final_pgeom[i]])

    max_x = patch_size[1] / 2
    max_y = patch_size[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    # results = []
    results = []
    for geom in final_pgeom:
        for ped_poly in geom.geoms:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)

            if lines.type != 'LineString':
                lines = ops.linemerge(lines)

            # same instance but not connected.
            if lines.type != 'LineString':
                ls = []
                for l in lines.geoms:
                    ls.append(np.array(l.coords))

                lines = np.concatenate(ls, axis=0)
                lines = LineString(lines)

            results.append(np.array(lines.coords))
    return results


if __name__ == '__main__':
    args = parse_args()
    for name in ['train', 'val', 'test']:
        create_av2_infos_mp(
            root_path=args.data_root,
            split=name,
            info_prefix='av2',
            dest_path=args.out_root,
            pc_range=args.pc_range,
            num_multithread=args.nproc,
            use_mixed=args.use_mixed,
            use_virtual=args.use_virtual,
            masked_elements=args.masked_elements)
