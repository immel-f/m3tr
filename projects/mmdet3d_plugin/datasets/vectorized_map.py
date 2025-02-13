import copy

import numpy as np
import torch
import numpy as np
import random

from shapely import affinity, ops
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString
from mmdet.datasets.pipelines import to_tensor

from pathlib import Path
import cv2

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords
    
class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates

    """
    def __init__(self, 
                 instance_line_list, 
                 instance_labels,
                 instance_label_names,
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,
                 patch_size=None,
                 code_size=2,
                 min_z=-5,
                 max_z=3,):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list
        self.code_size = code_size
        self.min_z = min_z
        self.max_z = max_z
        self.instance_labels = instance_labels
        self.instance_label_names = instance_label_names

        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno!")


    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno start_end_points!")

        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(
                                dtype=torch.float32)
        instance_se_points_tensor[:,0] = torch.clamp(instance_se_points_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,1] = torch.clamp(instance_se_points_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_se_points_tensor[:,2] = torch.clamp(instance_se_points_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_se_points_tensor[:,3] = torch.clamp(instance_se_points_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno bbox!")
            instance_bbox_array = np.zeros((0,4))
            instance_bbox_tensor = to_tensor(instance_bbox_array)
            return instance_bbox_tensor

        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(
                            dtype=torch.float32)
        instance_bbox_tensor[:,0] = torch.clamp(instance_bbox_tensor[:,0], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,1] = torch.clamp(instance_bbox_tensor[:,1], min=-self.max_y,max=self.max_y)
        instance_bbox_tensor[:,2] = torch.clamp(instance_bbox_tensor[:,2], min=-self.max_x,max=self.max_x)
        instance_bbox_tensor[:,3] = torch.clamp(instance_bbox_tensor[:,3], min=-self.max_y,max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno fixed_num_sampled_points!")
            instance_points_array = np.zeros((0,0,3))
            instance_points_tensor = to_tensor(instance_points_array)
            return instance_points_tensor

        instance_points_list = []
        for instance in self.instance_list:
            # instance_array = np.array(list(instance.coords))
            # interpolated_instance = interp_utils.interp_arc(t=self.fixed_num, points=instance_array)
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances])
            if instance.has_z:
                sampled_points = sampled_points.reshape(-1,3)
            else:
                sampled_points = sampled_points.reshape(-1,2)
            # import pdb;pdb.set_trace()
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)

        if instance.has_z:
            instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            instance_points_tensor[:,:,2] = torch.clamp(instance_points_tensor[:,:,2], min=self.min_z,max=self.max_z)
        else:
            instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,3]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno fixed_num_sampled_points_ambiguity!")

        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            if instance.has_z:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 3)
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)

        if instance.has_z:
            instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            instance_points_tensor[:,:,2] = torch.clamp(instance_points_tensor[:,:,2], min=self.min_z,max=self.max_z)
        else:
            instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        instance_points_tensor = instance_points_tensor if is_3d else instance_points_tensor[:,:,:2]
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def finstance_points_tensorixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno fixed_num_sampled_points_torch!")

        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0,2,1)
            sampled_pts = torch.nn.functional.interpolate(poly_pts,size=(self.fixed_num),mode='linear',align_corners=True)
            sampled_pts = sampled_pts.permute(0,2,1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list,dim=0)
        instance_points_tensor = instance_points_tensor.to(
                            dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x,max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y,max=self.max_y)
        instance_points_tensor[:,:,2] = torch.clamp(instance_points_tensor[:,:,2], min=self.min_z,max=self.max_z)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)
            shift_pts[:,:,2] = torch.clamp(shift_pts[:,:,2], min=self.min_z,max=self.max_z)


            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,shift_pts.shape[-1]], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)
            shift_pts[:,:,2] = torch.clamp(shift_pts[:,:,2], min=self.min_z,max=self.max_z)

            if not is_poly:
                padding = torch.full([shift_num-shift_pts.shape[0],pts_num,shift_pts.shape[-1]], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno shift_fixed_num_sampled_points_v2!")
            instance_points_array = np.zeros((0,0,self.code_size))
            instance_points_tensor = to_tensor(instance_points_array)
            return instance_points_tensor

        instances_list = []

        for idx, instance in enumerate(self.instance_list):
            instance_label_name = self.instance_label_names[idx]
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if instance_label_name == 'centerline':
                # import ipdb;ipdb.set_trace()
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                shift_pts_list.append(sampled_points)
            else:
                if is_poly:
                    pts_to_shift = poly_pts[:-1,:]
                    for shift_right_i in range(shift_num):
                        shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                        pts_to_concat = shift_pts[0]
                        pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                        shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                        shift_instance = LineString(shift_pts)
                        shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                        shift_pts_list.append(shift_sampled_points)
                    # import pdb;pdb.set_trace()
                else:
                    sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                    flip_sampled_points = np.flip(sampled_points, axis=0)
                    shift_pts_list.append(sampled_points)
                    shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            if instance.has_z:
                multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
                multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
                multi_shifts_pts_tensor[:,:,2] = torch.clamp(multi_shifts_pts_tensor[:,:,2], min=self.min_z,max=self.max_z)
            else:
                multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
                multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)

            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full([final_shift_num-multi_shifts_pts_tensor.shape[0],self.fixed_num,multi_shifts_pts_tensor.shape[-1]], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor[...,:self.code_size]

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        if len(self.instance_list) == 0:
            print("Warning: Empty instance list in map anno shift_fixed_num_sampled_points_v3!")

        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift,shift_right_i,axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat,axis=0)
                    shift_pts = np.concatenate((shift_pts,pts_to_concat),axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array([list(shift_instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, coords_num)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)
            
            multi_shifts_pts = np.stack(shift_pts_list,axis=0)
            shifts_num,_,_ = multi_shifts_pts.shape
            # import pdb;pdb.set_trace()
            if shifts_num > 2*final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index+shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts,flip1_shifts_pts),axis=0)
            
            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                            dtype=torch.float32)
            
            multi_shifts_pts_tensor[:,:,0] = torch.clamp(multi_shifts_pts_tensor[:,:,0], min=-self.max_x,max=self.max_x)
            multi_shifts_pts_tensor[:,:,1] = torch.clamp(multi_shifts_pts_tensor[:,:,1], min=-self.max_y,max=self.max_y)
            multi_shifts_pts_tensor[:,:,2] = torch.clamp(multi_shifts_pts_tensor[:,:,2], min=self.min_z,max=self.max_z)
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < 2*final_shift_num:
                padding = torch.full([final_shift_num*2-multi_shifts_pts_tensor.shape[0],self.fixed_num,multi_shifts_pts_tensor.shape[-1]], self.padding_value)
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor,padding],dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1,:]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i,0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num*2, pts_num, num_coords))
                tmp_shift_pts[:,:-1,:] = shift_pts
                tmp_shift_pts[:,-1,:] = shift_pts[:,0,:]
                shift_pts = tmp_shift_pts

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)
            shift_pts[:,:,2] = torch.clamp(shift_pts[:,:,2], min=self.min_z,max=self.max_z)


            if not is_poly:
                padding = torch.full([shift_num*2-shift_pts.shape[0],pts_num,shift_pts.shape[-1]], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i,0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list,dim=0)

            shift_pts[:,:,0] = torch.clamp(shift_pts[:,:,0], min=-self.max_x,max=self.max_x)
            shift_pts[:,:,1] = torch.clamp(shift_pts[:,:,1], min=-self.max_y,max=self.max_y)
            shift_pts[:,:,2] = torch.clamp(shift_pts[:,:,2], min=self.min_z,max=self.max_z)

            if not is_poly:
                padding = torch.full([fixed_num-shift_pts.shape[0],fixed_num,shift_pts.shape[-1]], self.padding_value)
                shift_pts = torch.cat([shift_pts,padding],dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(
                            dtype=torch.float32)
        return instances_tensor

    # @property
    # def polyline_points(self):
    #     """
    #     return [[x0,y0],[x1,y1],...]
    #     """
    #     assert len(self.instance_list) != 0
    #     for instance in self.instance_list:


class VectorizedLocalMap(object):
    # CLASS2LABEL = {
    #     'divider_dashed': 0,
    #     'divider_solid': 1,
    #     'divider_mixed': 2, 
    #     'divider_virtual': 3,
    #     'ped_crossing': 4,
    #     'boundary': 5, 
    #     'centerline': 6,
    #     'others': -1
    # }
    def __init__(self,
                 canvas_size, 
                 patch_size,
                 map_classes=['divider','ped_crossing','boundary'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000,
                 code_size=2,
                 min_z=-2,
                 max_z=2,
                 thickness=3,
                 aux_seg = dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32)):
        '''
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        '''
        super().__init__()

        self.vec_classes = map_classes

        self.class2label = {map_class: i for i, map_class in enumerate(map_classes)}

        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

        # for semantic mask
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.thickness = thickness
        self.scale_x = self.canvas_size[1] / self.patch_size[1]
        self.scale_y = self.canvas_size[0] / self.patch_size[0]
        # self.auxseg_use_sem = auxseg_use_sem
        self.aux_seg = aux_seg
        self.code_size =code_size

    def gen_vectorized_samples(self, map_annotation, example=None, feat_down_sample=32):
        '''
        use lidar2global to get gt map layers
        '''
        # avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)

        vectors = []
        for vec_class in self.vec_classes:
            instance_list = map_annotation[vec_class]
            if vec_class + '_map_data_idx' in map_annotation:
                instance_list_map_data_idx = map_annotation[vec_class + '_map_data_idx']
            else:
                instance_list_map_data_idx = np.array([-1 for el in instance_list])
            if vec_class + '_masked' in map_annotation and len(map_annotation[vec_class + '_masked']) > 0:
                instance_list_masked = map_annotation[vec_class + '_masked']
            else:
                instance_list_masked = np.array([True for el in instance_list])
            for i, instance in enumerate(instance_list):

                # if vec_class == 'boundary':
                #     print(vec_class)
                #     print(instance_list)
                #     print(instance_list_map_data_idx)
                #     print(instance_list_masked)

                if instance.shape[0] < 2:
                    if vec_class != 'ped_crossing':
                        print('class : {}, instance : {}, instance_list : {}'.format(vec_class, instance, instance_list))
                    continue
                
                vectors.append((LineString(np.array(instance)), self.class2label.get(vec_class, -1), vec_class, instance_list_map_data_idx[i], instance_list_masked[i]))

        gt_labels = []
        gt_label_names = []
        gt_instance = []
        gt_map_data_idx = []
        gt_masked = []
        # import ipdb;ipdb.set_trace()
        if self.aux_seg['use_aux_seg']:
            if self.aux_seg['seg_classes'] == 1:
                if self.aux_seg['bev_seg']:
                    gt_semantic_mask = np.zeros((1, self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
                else:
                    gt_semantic_mask = None
                # import ipdb;ipdb.set_trace()
                if self.aux_seg['pv_seg']:
                    num_cam  = len(example['img_metas'].data['pad_shape'])
                    img_shape = example['img_metas'].data['pad_shape'][0]
                    # import ipdb;ipdb.set_trace()
                    gt_pv_semantic_mask = np.zeros((num_cam, 1, img_shape[0] // feat_down_sample, img_shape[1] // feat_down_sample), dtype=np.uint8)
                    lidar2img = example['img_metas'].data['lidar2img']
                    scale_factor = np.eye(4)
                    scale_factor[0, 0] *= 1/32
                    scale_factor[1, 1] *= 1/32
                    lidar2feat = [scale_factor @ l2i for l2i in lidar2img]
                else:
                    gt_pv_semantic_mask = None
                for instance, instance_type, instance_type_name, map_data_idx, masked in vectors:
                    if instance_type != -1:
                        gt_instance.append(instance)
                        gt_labels.append(instance_type)
                        gt_label_names.append(instance_type_name)
                        gt_map_data_idx.append(map_data_idx)
                        gt_masked.append(masked)
                        if instance.geom_type == 'LineString':
                            if self.aux_seg['bev_seg']:
                                self.line_ego_to_mask(instance, gt_semantic_mask[0], color=1, thickness=self.thickness)
                            if self.aux_seg['pv_seg']:
                                for cam_index in range(num_cam):
                                    self.line_ego_to_pvmask(instance, gt_pv_semantic_mask[cam_index][0], lidar2feat[cam_index],color=1, thickness=self.aux_seg['pv_thickness'])
                        else:
                            print(instance.geom_type)
            else:
                if self.aux_seg['bev_seg']:
                    gt_semantic_mask = np.zeros((len(self.vec_classes), self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
                else:
                    gt_semantic_mask = None
                if self.aux_seg['pv_seg']:
                    num_cam  = len(example['img_metas'].data['pad_shape'])
                    gt_pv_semantic_mask = np.zeros((num_cam, len(self.vec_classes), img_shape[0] // feat_down_sample, img_shape[1] // feat_down_sample), dtype=np.uint8)
                    lidar2img = example['img_metas'].data['lidar2img']
                    scale_factor = np.eye(4)
                    scale_factor[0, 0] *= 1/32
                    scale_factor[1, 1] *= 1/32
                    lidar2feat = [scale_factor @ l2i for l2i in lidar2img]
                else:
                    gt_pv_semantic_mask = None
                for instance, instance_type, instance_type_name, map_data_idx, masked in vectors:
                    if instance_type != -1:
                        gt_instance.append(instance)
                        gt_labels.append(instance_type)
                        gt_label_names.append(instance_type_name)
                        gt_map_data_idx.append(map_data_idx)
                        gt_masked.append(masked)
                        if instance.geom_type == 'LineString':
                            if self.aux_seg['bev_seg']:
                                self.line_ego_to_mask(instance, gt_semantic_mask[instance_type], color=1, thickness=self.thickness)
                            if self.aux_seg['pv_seg']:
                                for cam_index in range(num_cam):
                                    self.line_ego_to_pvmask(instance, gt_pv_semantic_mask[cam_index][instance_type], lidar2feat[cam_index],color=1, thickness=self.aux_seg['pv_thickness'])
                        else:
                            print(instance.geom_type)
        else:
            for instance, instance_type, instance_type_name, map_data_idx, masked in vectors:
                if instance_type != -1:
                    gt_instance.append(instance)
                    gt_labels.append(instance_type)
                    gt_label_names.append(instance_type_name)
                    gt_map_data_idx.append(map_data_idx)
                    gt_masked.append(masked)
            gt_semantic_mask=None
            gt_pv_semantic_mask=None
        gt_instance = LiDARInstanceLines(gt_instance, gt_labels, gt_label_names, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size, code_size=self.code_size)


        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
            gt_semantic_mask=gt_semantic_mask,
            gt_pv_semantic_mask=gt_pv_semantic_mask,
            gt_map_data_idx=gt_map_data_idx,
            gt_masked=gt_masked,
        )
        return anns_results
    def line_ego_to_pvmask(self,
                          line_ego, 
                          mask, 
                          lidar2feat,
                          color=1, 
                          thickness=1,
                          default_z=-1.6):
        distances = np.linspace(0, line_ego.length, 200)
        coords = np.array([list(line_ego.interpolate(distance).coords) for distance in distances])
        if coords.size % 3 == 0:
            coords = coords.reshape(-1, 3)
            pts_num = coords.shape[0]
        else:
            coords = coords.reshape(-1, 2)
            pts_num = coords.shape[0]
            zeros = np.zeros((pts_num,1))
            zeros[:] = default_z
            coords = np.concatenate([coords,zeros], axis=1)
        ones = np.ones((pts_num,1))
        lidar_coords = np.concatenate([coords,ones], axis=1).transpose(1,0)
        pix_coords = perspective(lidar_coords, lidar2feat)
        cv2.polylines(mask, np.int32([pix_coords]), False, color=color, thickness=thickness)
        
    def line_ego_to_mask(self, 
                         line_ego, 
                         mask, 
                         color=1, 
                         thickness=3):
        ''' Rasterize a single line to mask.
        
        Args:
            line_ego (LineString): line
            mask (array): semantic mask to paint on
            color (int): positive label, default: 1
            thickness (int): thickness of rasterized lines, default: 3
        '''

        trans_x = self.canvas_size[1] / 2
        trans_y = self.canvas_size[0] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        # print(np.array(list(line_ego.coords), dtype=np.int32).shape)
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2
        
        cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)