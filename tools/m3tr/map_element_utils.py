import numpy as np
from copy import deepcopy
from collections import defaultdict
from scipy.spatial.transform import Rotation
import lanelet2
from lanelet2.core import (BasicPoint3d, Lanelet, LaneletMap,
                           LineString3d, Point2d, Point3d, getId)
from lanelet2.ml_converter import MapDataInterface, LineStringType, toPointMatrix


def generate_label_instances(ll2_map, ll2_map_routing_graph, e2g_translation, e2g_rotation, pc_range, use_mixed, use_virtual, merge_all_div_types, masked_elements, ignore_map_elevation=False):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_size = (patch_h, patch_w)
    map_pose = e2g_translation[:2]
    yaw, pitch, roll = Rotation.from_matrix(e2g_rotation).as_euler('zyx', degrees=False)
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])

    pos = BasicPoint3d(e2g_translation[0], e2g_translation[1], e2g_translation[2])                                        
    config = MapDataInterface.Configuration()
    config.submapExtentLongitudinal = patch_box[3] / 2
    config.submapExtentLateral = patch_box[2] / 2
    config.ignoreMapElevation = ignore_map_elevation
    mDataIf = MapDataInterface(ll2_map, config)            
    mDataIf.setCurrPosAndExtractSubmap(pos, yaw, pitch, roll) 
    lData = mDataIf.laneData(True)
    # print("Generated lData!")

    result_dict = {
        'divider': [],
        'divider_dashed': [],
        'divider_solid': [],
        'divider_mixed': [],
        'divider_virtual': [],
        'centerline': [],
        'boundary': [],
        'divider_map_data_idx': [],
        'divider_dashed_map_data_idx': [],
        'divider_solid_map_data_idx': [],
        'divider_mixed_map_data_idx': [],
        'divider_virtual_map_data_idx': [],
        'centerline_map_data_idx': [],
        'boundary_map_data_idx': [],
        'divider_masked_indices': [],
        'divider_dashed_masked_indices': [],
        'divider_solid_masked_indices': [],
        'divider_mixed_masked_indices': [],
        'divider_virtual_masked_indices': [],
        'centerline_masked_indices': [],
        'boundary_masked_indices': [],
        'divider_masked': [],
        'divider_dashed_masked': [],
        'divider_solid_masked': [],
        'divider_mixed_masked': [],
        'divider_virtual_masked': [],
        'centerline_masked': [],
        'boundary_masked': []
    }
    for i, div in enumerate(lData.compoundLaneDividers):
        if merge_all_div_types and div.type != LineStringType.Virtual:
            for lstring in div.cutInstance:
                result_dict['divider'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
                result_dict['divider_map_data_idx'].append(i)
        if div.type == LineStringType.Dashed:
            for lstring in div.cutInstance:
                result_dict['divider_dashed'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
                result_dict['divider_dashed_map_data_idx'].append(i)
        if div.type == LineStringType.Solid:
            for lstring in div.cutInstance:
                result_dict['divider_solid'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
                result_dict['divider_solid_map_data_idx'].append(i)
        if div.type == LineStringType.Mixed:
            for lstring in div.cutInstance:
                if use_mixed:
                    result_dict['divider_mixed'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
                    result_dict['divider_mixed_map_data_idx'].append(i)
                else:
                    dividers_solid.append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
                    result_dict['divider_solid_map_data_idx'].append(i)
        if div.type == LineStringType.Virtual:
            for lstring in div.cutInstance:
                result_dict['divider_virtual'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
                result_dict['divider_virtual_map_data_idx'].append(i)

    for i, bd in enumerate(lData.compoundRoadBorders):
        for lstring in bd.cutInstance:
            result_dict['boundary'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
            result_dict['boundary_map_data_idx'].append(i)

    for i, cline in enumerate(lData.compoundCenterlines):
        for lstring in cline.cutInstance:
            result_dict['centerline'].append(transform_line(toPointMatrix(lstring, False), e2g_translation, e2g_rotation))
            result_dict['centerline_map_data_idx'].append(i)

    if masked_elements is not None:
        return calc_masked_elements(ll2_map, ll2_map_routing_graph, lData, result_dict, (map_pose[0], map_pose[1], yaw), masked_elements, recurse_ego_lane_path_ll_ids=False), lData
    else:    
        return result_dict, lData


def transform_line(line_array, e2g_translation, e2g_rotation, use_only_yaw=False, set_z_to_zero=False):
    #print('BEFORE: ')
    #print(line_array)

    new_line_pts_trans = line_array - e2g_translation[np.newaxis, :]
    if use_only_yaw:
        yaw, pitch, roll = Rotation.from_matrix(e2g_rotation).as_euler('zyx', degrees=False)
        new_line_pts_egoframe = (Rotation.from_euler('z', yaw, degrees=False).as_matrix().T @ new_line_pts_trans.T).T
    else:
        new_line_pts_egoframe = (e2g_rotation.T @ new_line_pts_trans.T).T
    # line = LineString(new_line_pts_egoframe) 

    #print('AFTER: ')
    #print(new_line_pts_egoframe)

    if set_z_to_zero:
        new_line_pts_egoframe[:, 2] = 0

    return new_line_pts_egoframe


def get_ll_associated_indices(ll_id, cpdInstanceList):
    asso_indices = []
    for i, cpdInst in enumerate(cpdInstanceList):
        for indInst in cpdInst.features:
            for ll_id_inst in indInst.laneletIDs:
                if ll_id == ll_id_inst:
                    asso_indices.append(i)
    return asso_indices

def calc_masked_elements_for_ll(result_dict, lData, ll_id, include_boundaries=True):
    divider_masked_indices = get_ll_associated_indices(ll_id, lData.compoundLaneDividers)
    centerline_masked_indices = get_ll_associated_indices(ll_id, lData.compoundCenterlines)
    boundary_masked_indices = get_ll_associated_indices(ll_id, lData.compoundRoadBorders)

    for i, map_data_idx in enumerate(result_dict['divider_dashed_map_data_idx']):
        if map_data_idx in divider_masked_indices and i not in result_dict['divider_dashed_masked_indices']:
            result_dict['divider_dashed_masked_indices'].append(i)
    
    for i, map_data_idx in enumerate(result_dict['divider_solid_map_data_idx']):
        if map_data_idx in divider_masked_indices and i not in result_dict['divider_solid_masked_indices']:
            result_dict['divider_solid_masked_indices'].append(i)
    
    for i, map_data_idx in enumerate(result_dict['divider_mixed_map_data_idx']):
        if map_data_idx in divider_masked_indices and i not in result_dict['divider_mixed_masked_indices']:
            result_dict['divider_mixed_masked_indices'].append(i)
    
    for i, map_data_idx in enumerate(result_dict['divider_virtual_map_data_idx']):
        if map_data_idx in divider_masked_indices and i not in result_dict['divider_virtual_masked_indices']:
            result_dict['divider_virtual_masked_indices'].append(i)
    
    for i, map_data_idx in enumerate(result_dict['centerline_map_data_idx']):
        if map_data_idx in centerline_masked_indices and i not in result_dict['centerline_masked_indices']:
            result_dict['centerline_masked_indices'].append(i)

    if include_boundaries:
        for i, map_data_idx in enumerate(result_dict['boundary_map_data_idx']):
            if map_data_idx in boundary_masked_indices and i not in result_dict['boundary_masked_indices']:
                result_dict['boundary_masked_indices'].append(i)


def result_dict_indices_to_masks(result_dict, include_boundaries=True):
    divider_dashed_mask = np.array([False for i in range(0, len(result_dict['divider_dashed']))])
    divider_dashed_mask[result_dict['divider_dashed_masked_indices']] = True
    result_dict['divider_dashed_masked'] = divider_dashed_mask

    divider_solid_mask = np.array([False for i in range(0, len(result_dict['divider_solid']))])
    divider_solid_mask[result_dict['divider_solid_masked_indices']] = True
    result_dict['divider_solid_masked'] = divider_solid_mask

    divider_mixed_mask = np.array([False for i in range(0, len(result_dict['divider_mixed']))])
    divider_mixed_mask[result_dict['divider_mixed_masked_indices']] = True
    result_dict['divider_mixed_masked'] = divider_mixed_mask

    divider_virtual_mask = np.array([False for i in range(0, len(result_dict['divider_virtual']))])
    divider_virtual_mask[result_dict['divider_virtual_masked_indices']] = True
    result_dict['divider_virtual_masked'] = divider_virtual_mask

    centerline_mask = np.array([False for i in range(0, len(result_dict['centerline']))])
    centerline_mask[result_dict['centerline_masked_indices']] = True
    result_dict['centerline_masked'] = centerline_mask

    if include_boundaries:
        boundary_mask = np.array([False for i in range(0, len(result_dict['boundary']))])
        boundary_mask[result_dict['boundary_masked_indices']] = True
        result_dict['boundary_masked'] = boundary_mask

def get_path_ll_ids(ll_id, lData, existing_dict=None, recurse=True):
    centerline_indices = get_ll_associated_indices(ll_id, lData.compoundCenterlines)
    if existing_dict is None:
        path_ll_ids = {}
    else:
        path_ll_ids = existing_dict
    for idx in centerline_indices:
        cpdInst = lData.compoundCenterlines[idx]
        for indInst in cpdInst.features:
            for ll_id_inst in indInst.laneletIDs:
                path_ll_ids[ll_id_inst] = True
    
    if recurse:
        path_ll_ids_copy = deepcopy(path_ll_ids)
        for ll_id in path_ll_ids_copy:
            get_path_ll_ids(ll_id, lData, path_ll_ids, False)
    return path_ll_ids


def get_road_ll_ids(ego_lanelet, ll2_map_routing_graph, ll2_map):
    road_ll_ids = []
    road_ll_ids.append(ego_lanelet.id)

    if ll2_map_routing_graph is not None:
        graph = ll2_map_routing_graph
    else:
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                              lanelet2.traffic_rules.Participants.Vehicle)
        graph = lanelet2.routing.RoutingGraph(ll2_map, traffic_rules)

    left_ll = graph.left(ego_lanelet)
    adj_left_ll = graph.adjacentLeft(ego_lanelet)

    while left_ll is not None or adj_left_ll is not None:
        if left_ll is not None:
            road_ll_ids.append(left_ll.id)
            adj_left_ll = graph.adjacentLeft(left_ll)
            left_ll = graph.left(left_ll)
        elif adj_left_ll is not None:
            road_ll_ids.append(adj_left_ll.id)
            left_ll = graph.left(adj_left_ll)
            adj_left_ll = graph.adjacentLeft(adj_left_ll)

    right_ll = graph.right(ego_lanelet)
    adj_right_ll = graph.adjacentRight(ego_lanelet)

    while right_ll is not None or adj_right_ll is not None:
        if right_ll is not None:
            road_ll_ids.append(right_ll.id)
            adj_right_ll = graph.adjacentRight(right_ll)
            right_ll = graph.right(right_ll)
        elif adj_right_ll is not None:
            road_ll_ids.append(adj_right_ll.id)
            right_ll = graph.right(adj_right_ll)
            adj_right_ll = graph.adjacentRight(adj_right_ll)
    
    return road_ll_ids


def calc_masked_elements(ll2_map, ll2_map_routing_graph, lData, result_dict, map_pose, masked_elements, include_boundaries=True, recurse_ego_lane_path_ll_ids=True):
    # TODO: Mask for all lanelets in path for ego lane and use orientation for lanelet2_matching to fix wrong association at crossings
    if masked_elements[0] == 'ego_lane':  
        ego_obj2d = lanelet2.matching.ObjectWithCovariance2d(1, lanelet2.matching.Pose2d(map_pose[0], map_pose[1], map_pose[2]), [], lanelet2.matching.PositionCovariance2d(1, 1, 0.1), 2)

        # print([el.lanelet.id for el in lanelet2.matching.getDeterministicMatches(ll2_map, ego_obj2d, 3.5)])
        ego_lanelet_matches = lanelet2.matching.getProbabilisticMatches(ll2_map, ego_obj2d, 3.0)

        if len(ego_lanelet_matches) > 0:
            ego_lanelet_id = ego_lanelet_matches[0].lanelet.id
        else:
            print("WARNING: No matching lanelet found for ego lane masking, skipping masking for this sample...")
            return result_dict
        
        path_ll_ids = get_path_ll_ids(ego_lanelet_id, lData, existing_dict=None, recurse=recurse_ego_lane_path_ll_ids)
        for ll_id in path_ll_ids.keys():
            calc_masked_elements_for_ll(result_dict, lData, ll_id, include_boundaries=include_boundaries)

    elif masked_elements[0] == 'ego_road':
        
        ego_obj2d = lanelet2.matching.ObjectWithCovariance2d(1, lanelet2.matching.Pose2d(map_pose[0], map_pose[1], map_pose[2]), [], lanelet2.matching.PositionCovariance2d(1, 1, 0.1), 2)
        ego_lanelet_matches = lanelet2.matching.getProbabilisticMatches(ll2_map, ego_obj2d, 3.0)

        if len(ego_lanelet_matches) > 0:
            ego_lanelet = ego_lanelet_matches[0].lanelet
        else:
            print("WARNING: No matching lanelet found for ego road masking, skipping masking for this sample...")
            return result_dict

        road_ll_ids = get_road_ll_ids(ego_lanelet, ll2_map_routing_graph, ll2_map)

        # get all lanelets that "cross" road paths and then also get all lanelets that "cross" road lanelets
        # this is neccessary for crossings since there the road neighbors of the current lanelet may not be there 
        path_ll_ids = {}
        for ll_id in road_ll_ids:
            get_path_ll_ids(ll_id, lData, path_ll_ids, recurse=recurse_ego_lane_path_ll_ids)

        path_ll_ids_tmp = deepcopy(path_ll_ids) 
        for ll_id_path in path_ll_ids_tmp.keys():
            road_ll_ids_path_ll = get_road_ll_ids(ll2_map.laneletLayer[ll_id_path], ll2_map_routing_graph, ll2_map)
            for ll_id_path_ll in road_ll_ids_path_ll:
                path_ll_ids[ll_id_path_ll] = True
        for ll_id_path in path_ll_ids.keys():
            calc_masked_elements_for_ll(result_dict, lData, ll_id_path, include_boundaries=include_boundaries)

    else:
        for el_type in masked_elements:
            if el_type in result_dict:
                result_dict[el_type + '_masked_indices'] = list(range(0, len(result_dict[el_type])))
            elif el_type != 'boundary' and el_type != 'ped_crossing':
                raise ValueError("Unknown element type " + el_type)

    result_dict['divider_dashed_masked_indices'] = np.array(result_dict['divider_dashed_masked_indices'], dtype=int)
    result_dict['divider_solid_masked_indices'] = np.array(result_dict['divider_solid_masked_indices'], dtype=int)
    result_dict['divider_mixed_masked_indices'] = np.array(result_dict['divider_mixed_masked_indices'], dtype=int)
    result_dict['divider_virtual_masked_indices'] = np.array(result_dict['divider_virtual_masked_indices'], dtype=int)
    result_dict['centerline_masked_indices'] = np.array(result_dict['centerline_masked_indices'], dtype=int)

    if include_boundaries:
        result_dict['boundary_masked_indices'] = np.array(result_dict['boundary_masked_indices'], dtype=int)

    result_dict_indices_to_masks(result_dict, include_boundaries=include_boundaries)

    return result_dict