import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner import force_fp32, auto_fp16


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

@TRANSFORMER.register_module()
class BEVInstTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 num_proposal=0,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(BEVInstTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.num_proposal = num_proposal
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        # self.refpos_embedding = MLP(3, self.embed_dims, self.embed_dims, 3)
        self.refpos_embedding = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(m, BEVInstCrossAtten):
                m.init_weight()
        xavier_init(self.refpos_embedding, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats,
                query_embed,
                reg_branches=None,
                priors=None,
                **kwargs):
        if query_embed is not None:
            query = query_embed['query_embed']
            bboxes = query_embed['bboxes']
            bs = mlvl_feats[0].size(0)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            bboxes = bboxes.unsqueeze(0).expand(bs, -1, -1)
            reference_points = bboxes[..., [0,1,4]].sigmoid()
            query_pos = self.refpos_embedding(reference_points)
        
        if priors is not None:
            prev_query_feat = priors['query_embed']
            prev_bboxes = priors['bboxes']
            prev_reference_points = prev_bboxes[..., [0,1,4]].sigmoid()
            prev_query_pos = self.refpos_embedding(prev_reference_points)
            
            if query_embed is not None:
                query = torch.cat([prev_query_feat, query], dim=1)
                query_pos = torch.cat([prev_query_pos, query_pos], dim=1)
                bboxes = torch.cat([prev_bboxes, bboxes], dim=1)
                reference_points = torch.cat(
                    [prev_reference_points, reference_points], dim=1)
            else:
                query = prev_query_feat
                query_pos = prev_query_pos
                bboxes = prev_bboxes
                reference_points = prev_reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_bboxes = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            bboxes=bboxes,
            **kwargs)
        
        return inter_states, inter_bboxes

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVInstTransformerDecoder(TransformerLayerSequence):
    def __init__(self, *args, transformerlayers_twin=None, return_intermediate=False,
                 multi_offset=False, normalize_yaw=False, **kwargs):
        super(BEVInstTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.multi_offset = multi_offset
        self.normalize_yaw = normalize_yaw

        if transformerlayers_twin is not None:
            import copy
            from mmcv.cnn.bricks.transformer import build_transformer_layer
            from mmcv.runner.base_module import ModuleList
            if isinstance(transformerlayers_twin, dict):
                transformerlayers_twin = [
                    copy.deepcopy(transformerlayers_twin) for _ in range(self.num_layers)
                ]
            else:
                assert isinstance(transformerlayers_twin, list) and \
                    len(transformerlayers_twin) == self.num_layers
            self.layers_twin = ModuleList()
            for i in range(self.num_layers):
                self.layers_twin.append(build_transformer_layer(transformerlayers_twin[i]))
            self.num_layers = self.num_layers*2
        else:
            self.layers_twin = None

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                bboxes=None,
                **kwargs):
        output = query
        velocity = bboxes[..., [8,9]]
        intermediate = []
        intermediate_bboxes = []
        for i, layer in enumerate(self.layers):
            lid = 2*i if self.layers_twin is not None else i
            reference_points_input = reference_points
            velocity_input = velocity
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                velocity=velocity_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                # bboxes = 0:x, 1:y, 2:l, 3:w, 4:z, 5:h, 6:sin, 7:cos, 8:vx, 9:vy
                if self.multi_offset:
                    bboxes[..., :6] = bboxes[..., :6] + tmp[..., :6]
                    if self.normalize_yaw:
                        bboxes[..., 6:8] = F.normalize(tmp[..., 6:8], dim=-1)
                    else:
                        bboxes[..., 6:8] = tmp[..., 6:8]
                    bboxes[..., 8:] = tmp[..., 8:] / 0.5
                else:
                    bboxes = bboxes + tmp
                new_reference_points = bboxes[..., [0,1,4]].sigmoid()
                reference_points = new_reference_points.detach()
                velocity = bboxes[..., [8,9]].detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_bboxes.append(bboxes)

            # ---------------------------------------------------------
            if self.layers_twin is not None:
                reference_points_input = reference_points
                velocity_input = velocity
                output = self.layers_twin[i](
                    output,
                    *args,
                    reference_points=reference_points_input,
                    velocity=velocity_input,
                    **kwargs)
                output = output.permute(1, 0, 2)

                if reg_branches is not None:
                    tmp = reg_branches[2*i+1](output)
                    # bboxes = 0:x, 1:y, 2:l, 3:w, 4:z, 5:h, 6:sin, 7:cos, 8:vx, 9:vy
                    if self.multi_offset:
                        bboxes[..., :6] = bboxes[..., :6] + tmp[..., :6]
                        if self.normalize_yaw:
                            bboxes[..., 6:8] = F.normalize(tmp[..., 6:8], dim=-1)
                        else:
                            bboxes[..., 6:8] = tmp[..., 6:8]
                        bboxes[..., 8:] = tmp[..., 8:] / 0.5
                    else:
                        bboxes = bboxes + tmp
                    new_reference_points = bboxes[..., [0,1,4]].sigmoid()
                    reference_points = new_reference_points.detach()
                    velocity = bboxes[..., [8,9]].detach()

                output = output.permute(1, 0, 2)
                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_bboxes.append(bboxes)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_bboxes)

        return output, bboxes

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim = None, dropout = 0.):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

@ATTENTION.register_module()
class BEVInstCrossAtten(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=5,
                 num_cams=6,
                 num_frames=9,
                 temporal_weight=0.9**(0.5*10),
                 im2col_step=64,
                 pc_range=None,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 use_vel=True,
                 batch_first=False):
        super(BEVInstCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.use_vel=use_vel
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            import warnings
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.num_frames = num_frames
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams*num_frames*num_levels*num_points)
        if self.num_frames > 1:
            self.temporal_weight = temporal_weight
            self.temporal_net = FeedForward(embed_dims*2, embed_dims*4, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
      
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                velocity=None,
                spatial_shapes=None,
                level_start_index=None,
                
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_frames, self.num_points, self.num_levels)
        
        reference_points_3d, output, mask = feature_sampling_4d(
            value, reference_points, velocity, self.pc_range, kwargs['img_metas'], use_vel=self.use_vel)
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-3) # scale view
        output = output.permute(3, 4, 2, 0, 1)
        # B C N T K -> T K N B C -> K N B C
        output_t = output[0]
        if self.num_frames > 1:
            for i in range(1, self.num_frames):
                output_aux = output[i] * self.temporal_weight
                output_t = torch.cat([output_t, output_aux], -1) 
                output_t = self.temporal_net(output_t)
        output = output_t.sum(0) # keypoint
        
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat
    
# @force_fp32(apply_to=('reference_points', 'img_metas', 'velocity'))
def feature_sampling_4d(mlvl_feats, reference_points, velocity, pc_range, img_metas, use_vel=True):
    rt2img = torch.stack([img_meta['rt2img'] for img_meta in img_metas])
    num_frame = rt2img.shape[2]
    rt2img = reference_points.new_tensor(rt2img) # (B, N, T, 4, 4)
    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    velocity = velocity.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    B, num_query = reference_points.size()[:2]

    reference_points = reference_points.unsqueeze(2).repeat(1, 1, num_frame, 1)
    if use_vel:
        for t in range(num_frame):
            reference_points[..., t, :2] = reference_points[..., t, :2] - 0.5*t*velocity
    reference_points = reference_points.permute(0, 2, 1, 3)

    if 'bda' in img_metas[0].keys():
        bda_inv = torch.stack([img_meta['bda'].inverse() for img_meta in img_metas])
        bda_inv = bda_inv.view(B, 1, 1, 3, 3).repeat(1, num_frame, num_query, 1, 1)
        reference_points = torch.matmul(bda_inv, reference_points.unsqueeze(-1)).squeeze(-1)
    # reference_points (B, num_frame, num_queries, 4)
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    num_cam = rt2img.size(1)
    reference_points = reference_points[:, None].repeat(1, num_cam, 1, 1, 1).unsqueeze(-2)
    rt2img = rt2img.view(B, num_cam, num_frame, 1, 4, 4).repeat(1, 1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(reference_points, rt2img).squeeze(-2)
    eps = 1e-5
    mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
    post_rot_xy_T = torch.stack([img_meta['post_rot_xy_T'] for img_meta in img_metas])
    post_tran_xy = torch.stack([img_meta['post_tran_xy'] for img_meta in img_metas])
    post_rot_xy_T = post_rot_xy_T.view(B, num_cam, num_frame, 1, 2, 2).repeat(1, 1, 1, num_query, 1, 1)
    post_tran_xy = post_tran_xy.view(B, num_cam, num_frame, 1, 2)
    reference_points_cam = torch.matmul(reference_points_cam.unsqueeze(-2), 
        post_rot_xy_T).squeeze(-2) + post_tran_xy

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0]
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) 
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) 
                 & (reference_points_cam[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, num_frame, 1, num_query, 1, 1).permute(0, 3, 4, 1, 2, 5, 6) # B 1 900 6 9 1 1
    mask = torch.nan_to_num(mask)
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, T, C, H, W = feat.size()
        feat = feat.contiguous().view(B*N*T, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N*T, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = sampled_feat.view(B, N, T, C, num_query, 1).permute(0, 3, 4, 1, 2, 5)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    # 1 为sample的组数  即keypoint
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, num_frame, 1, len(mlvl_feats))
    return reference_points_3d, sampled_feats, mask


@ATTENTION.register_module()
class BEVInstBEVCrossAtten(BEVInstCrossAtten):
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_feats=None,
                **kwargs):

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_frames, self.num_points, self.num_levels)
        
        reference_points_3d, output, mask = feature_sampling_bev(
            bev_feats[0], reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)

        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-3) # scale view
        output = output.permute(3, 4, 2, 0, 1)
        # B C N T K -> T K N B C -> K N B C
        output_t = output[-1]
        if self.num_frames > 1:
            for i in range(2, self.num_frames+1):
                output_t = torch.cat([output[-i], output_t], -1) 
                output_t = self.temporal_net(output_t)
        output = output_t.sum(0) # keypoint
        
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)

        return self.dropout(output) + inp_residual + pos_feat

# @force_fp32(apply_to=('reference_points', 'img_metas'))
def feature_sampling_bev(bev_feats, reference_points, pc_range, img_metas):
    out_size_factor = img_metas[0]['out_size_factor']
    voxel_size = img_metas[0]['voxel_size']
    B, C, H, W = bev_feats.shape
    num_cam = 1
    num_frame = 1
    num_level = 1

    reference_points = reference_points.clone()
    reference_points_3d = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
    
    num_query = reference_points.size()[1]
    reference_points_bev = reference_points[..., :2]
    
    reference_points_bev[..., 0] = (reference_points_bev[..., 0] - pc_range[0]
        ) / (out_size_factor * voxel_size[0])
    reference_points_bev[..., 1] = (reference_points_bev[..., 1] - pc_range[1]
        ) / (out_size_factor * voxel_size[1])
    reference_points_bev[..., 0] /= W
    reference_points_bev[..., 1] /= H
    reference_points_bev = (reference_points_bev - 0.5) * 2

    mask = ((reference_points_bev[..., 0:1] > -1.0) 
            & (reference_points_bev[..., 0:1] < 1.0) 
            & (reference_points_bev[..., 1:2] > -1.0) 
            & (reference_points_bev[..., 1:2] < 1.0))
    mask = mask.view(B, num_cam, num_frame, 1, num_query, 1, 1).permute(0, 3, 4, 1, 2, 5, 6) # B 1 900 6 9 1 1
    mask = torch.nan_to_num(mask)
    sampled_feats = F.grid_sample(bev_feats, reference_points_bev.unsqueeze(1)).squeeze(2).detach().clone()
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam, num_frame, 1, num_level)
    return reference_points_3d, sampled_feats, mask
