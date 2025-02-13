import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from .builder import build_fuser, FUSERS
from typing import List

def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    new_pts[...,2:3] = (pts[...,2:3]*(pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_z = pc_range[5]-pc_range[2]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    new_pts[...,2:3] = pts[...,2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h,patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts

def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))



@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 z_cfg=dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 feat_down_sample_indice=-1,
                 **kwargs):
        super(MapTRPerceptionTransformer, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        # self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.use_attn_bev = 'BEVFormerEncoder' in encoder['type']
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg=z_cfg
        self.init_layers()
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) if not self.z_cfg['gt_z_flag'] \
                            else nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def attn_bev_encode(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        ret_dict = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            mlvl_feats=mlvl_feats,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return ret_dict

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        # import ipdb;ipdb.set_trace()
        # assert len(mlvl_feats) == 1, 'Currently we only use last single level feat in LSS'
        # import ipdb;ipdb.set_trace()


        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs['img_metas']
        encoder_outputdict = self.encoder(images,img_metas)
        bev_embed = encoder_outputdict['bev']
        depth = encoder_outputdict['depth']
        bs, c, _,_ = bev_embed.shape
        bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        if self.use_attn_bev:
            ret_dict = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs)
            bev_embed = ret_dict['bev']
            depth = ret_dict['depth']
        else:
            ret_dict = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs)
            bev_embed = ret_dict['bev']
            depth = ret_dict['depth']
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h,bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0,2,1).contiguous()
            bev_embed = fused_bev
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        return feat_flatten, spatial_shapes, level_start_index
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                gt_pts_list_normalized=None, 
                gt_labels=None, 
                gt_masked=None,
                gt_object_query_embed=None,
                mask_assign_gt_one2many=False,
                use_gt_pos_queries=True,
                use_gt_ref_points=True,
                use_learned_gt_queries=True,
                num_vec_one2many=300,
                k_one2many=6,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        bev_embed = ouput_dic['bev']
        depth = ouput_dic['depth']
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)

        gt_query_pos, gt_query = torch.split(
            gt_object_query_embed, self.embed_dims, dim=1)

        # print(query.shape)

        query_with_map = query.clone()
        query_pos_with_map = query_pos.clone()

        if gt_object_query_embed is not None:
            for i, gt_mask in enumerate(gt_masked):
                query_mask = torch.logical_not(gt_mask)
                if torch.count_nonzero(query_mask) > 0:
                    gt_pts = gt_pts_list_normalized[i][query_mask]
                    gt_pts = inverse_sigmoid(gt_pts)
                    gt_num_el, gt_num_pts, gt_pts_dim = gt_pts.shape

                    # print(gt_pts.shape)
                    # print(gt_labels[i][query_mask].shape)

                    gt_pts_flattened = gt_pts.view(gt_num_el*gt_num_pts, gt_pts_dim)
                    gt_labels_query = gt_labels[i][query_mask].repeat_interleave(gt_num_pts)
                    gt_labels_query = nn.functional.one_hot(gt_labels_query)

                    # print(gt_labels_query.shape)

                    gt_pts_query = torch.zeros(gt_num_el*gt_num_pts, self.embed_dims, device=gt_pts.device)
                    gt_pts_query[:, 0:gt_pts_dim] = gt_pts_flattened
                    gt_pts_query[:, gt_pts_dim:(gt_pts_dim+gt_labels_query.shape[-1])] = gt_labels_query

                    # hacky: don't add gt elements that would exceed num_vec
                    query_slice = min(gt_num_el*gt_num_pts, gt_query.shape[0])
                    if use_learned_gt_queries:
                        gt_pts_query = gt_pts_query[0:query_slice, :] + gt_query[0:query_slice, :]
                        gt_pts_query_pos = gt_pts_query[0:query_slice, :] + gt_query_pos[0:query_slice, :]
                    else:
                        gt_pts_query = gt_pts_query[0:query_slice, :]
                        gt_pts_query_pos = gt_pts_query[0:query_slice, :]
                    # print("GT MAP EL QUERY:")
                    # print(gt_pts_query.view(gt_num_el, gt_num_pts, self.embed_dims))

                    query_pos_with_map[i, 0:query_slice] = gt_pts_query_pos
                    query_with_map[i, 0:query_slice] = gt_pts_query

                    
                    if mask_assign_gt_one2many and query.shape[1] > num_vec_one2many*gt_num_pts: #check if we are in train or test mode, hacky
                        one2many_start = query.shape[1] - num_vec_one2many*gt_num_pts
                        gt_query_len = k_one2many*query_slice

                        if gt_query_len > query.shape[1] - one2many_start:
                            gt_query_len = (k_one2many-1)*query_slice
                            repeat_num = k_one2many-1
                        else:
                            repeat_num = k_one2many

                        query_pos_with_map[i, one2many_start:(one2many_start + gt_query_len)] = gt_pts_query_pos.repeat(repeat_num, 1)
                        query_with_map[i, one2many_start:(one2many_start + gt_query_len)] = gt_pts_query.repeat(repeat_num, 1)

            query = query_with_map
            if use_gt_pos_queries:
                query_pos = query_pos_with_map

        reference_points = self.reference_points(query_pos)

        reference_points_with_gt = reference_points.clone()
        if gt_object_query_embed is not None:
            for i, gt_mask in enumerate(gt_masked):

                query_mask = torch.logical_not(gt_mask)

                if torch.count_nonzero(query_mask) > 0:

                    gt_pts = gt_pts_list_normalized[i][query_mask]
                    gt_pts = inverse_sigmoid(gt_pts)
                    gt_num_el, gt_num_pts, gt_pts_dim = gt_pts.shape
                    query_slice = min(gt_num_el*gt_num_pts, gt_query.shape[0])

                    gt_pts_flattened = gt_pts.view(gt_num_el*gt_num_pts, gt_pts_dim)
                    reference_points_with_gt[i, 0:query_slice] = gt_pts_flattened[0:query_slice]

                    if mask_assign_gt_one2many and reference_points_with_gt.shape[1] > num_vec_one2many*gt_num_pts: #check if we are in train or test mode, hacky
                        one2many_start = reference_points_with_gt.shape[1] - num_vec_one2many*gt_num_pts
                        gt_query_len = k_one2many*query_slice

                        if gt_query_len > query.shape[1] - one2many_start:
                            gt_query_len = (k_one2many-1)*query_slice
                            repeat_num = k_one2many-1
                        else:
                            repeat_num = k_one2many

                        reference_points_with_gt[i, one2many_start:(one2many_start + gt_query_len)] = gt_pts_flattened[0:query_slice].repeat(repeat_num, 1)
            
            if use_gt_ref_points:
                reference_points = reference_points_with_gt

        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points    

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        feat_flatten, feat_spatial_shapes, feat_level_start_index \
            = self.format_feats(mlvl_feats)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            mlvl_feats=mlvl_feats,
            feat_flatten=feat_flatten,
            feat_spatial_shapes=feat_spatial_shapes,
            feat_level_start_index=feat_level_start_index,
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, depth, inter_states, init_reference_out, inter_references_out
