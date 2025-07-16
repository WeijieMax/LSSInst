import torch
import torch.nn.functional as F
from .. import builder
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.util import normalize_bbox
from mmdet.models import DETECTORS
from mmdet.models.utils import build_transformer
from .solofusion import SOLOFusion
from .. import builder
from mmcv.runner import force_fp32, auto_fp16
import torch.nn as nn
import math
from mmcv.cnn import xavier_init, constant_init

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ =\
        sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()

@DETECTORS.register_module()
class SOLOInstDeform(SOLOFusion):
    def __init__(self,
                 post_img_neck=None,
                 post_image_encoder=None,
                 post_pts_bbox_head=None,
                 num_frame=1,
                 num_post_frame=1,
                 post_test=True,
                 freeze_bev=False,
                 deform_sample=False,
                 embed_dims=256,
                 num_heads=8,
                 num_points=4,
                 num_levels=1,
                 num_bev_queue=1,
                 dropout=0.1,
                 **kwargs):
        super(SOLOInstDeform, self).__init__(**kwargs)
        if post_pts_bbox_head:
            assert post_img_neck is not None
            self.post_img_neck = builder.build_neck(post_img_neck)
            if post_image_encoder is not None:
                self.post_image_encoder = build_transformer(post_image_encoder)
            else:
                self.post_image_encoder = None
            self.post_pts_bbox_head = builder.build_head(post_pts_bbox_head)
            self.num_proposal = self.post_pts_bbox_head.transformer.num_proposal
            self.with_prior_grad = self.post_pts_bbox_head.with_prior_grad
        else:
            self.post_pts_bbox_head = None
            self.num_proposal = None
        self.num_frame = num_frame
        self.num_post_frame = num_post_frame
        self.post_test = post_test
        self.freeze_bev = freeze_bev
        
        self.deform_sample = deform_sample
        if self.deform_sample:
            self.embed_dims = embed_dims
            self.num_heads = num_heads
            self.num_points = num_points
            self.num_levels = num_levels
            self.num_bev_queue = num_bev_queue
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads  * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims,
                                            num_heads *  num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.dropout = nn.Dropout(dropout)
            self.init_weights()
            
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, 1, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True    
        
    @auto_fp16()
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        
        neck_feats = self.img_neck(backbone_feats)
        if isinstance(neck_feats, list):
            assert len(neck_feats) == 1 # SECONDFPN returns a length-one list
            neck_feats = neck_feats[0]
            
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        if self.do_history_stereo_fusion:
            backbone_feats_detached = [tmp.detach() for tmp in backbone_feats]
            stereo_feats = self.stereo_neck(backbone_feats_detached)
            if isinstance(stereo_feats, list):
                assert len(stereo_feats) == 1 # SECONDFPN returns a trivial list
                stereo_feats = stereo_feats[0]
            stereo_feats = F.normalize(stereo_feats, dim=1, eps=self.img_view_transformer.stereo_eps)
            return neck_feats, stereo_feats.view(B, N, *stereo_feats.shape[1:]), backbone_feats
        else:
            return neck_feats, None, backbone_feats

    def extract_img_feat(self, img, img_metas, gt_bboxes_3d=None):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        imgs = inputs[0].view(B, N, 1, 3, H, W)
        imgs = torch.split(imgs, 1, dim=2)
        imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W
  
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]

        extra = [rots.view(B, 1, N, 3, 3),
                 trans.view(B, 1, N, 3),
                 intrins.view(B, 1, N, 3, 3),
                 post_rots.view(B, 1, N, 3, 3),
                 post_trans.view(B, 1, N, 3)]
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra] # each B x N x 3 (x 3)
        rots, trans, intrins, post_rots, post_trans = extra

        bev_feat_list = []
        depth_digit_list = []
        geom_list = []
            
        curr_img_encoder_feats, curr_stereo_feats, backbone_feats = self.image_encoder(imgs[0])
        if not self.do_history_stereo_fusion:
            bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
        else:
            prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam = \
                self.process_stereo_before_fusion(curr_stereo_feats, img_metas, rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
            bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0],
                curr_stereo_feats, prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam)
            self.process_stereo_for_next_timestep(img_metas, curr_stereo_feats, curr_global2img, curr_img_forward_aug)

        bev_feat = self.pre_process_net(bev_feat)[0] # singleton list
        bev_feat = self.embed(bev_feat)

        # Fuse History
        if self.do_history:
            bev_feat = self.fuse_history(bev_feat, img_metas)

        x = self.bev_encoder(bev_feat)

        return x, depth_digit, backbone_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        losses = dict()

        img_feats, depth, backbone_feats = self.extract_img_feat(img_inputs, img_metas)

        if not self.freeze_bev:
            # If we're training depth...
            depth_gt = img_inputs[-1] 
            loss_depth = self.get_depth_loss(depth_gt, depth)
            losses['loss_depth'] = loss_depth
            
            # Get box losses
            bbox_outs = self.pts_bbox_head(img_feats)
            losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
            losses.update(losses_pts)
        else:
            losses = dict()

        if self.post_pts_bbox_head:
            bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=False)
            outs = self.forward_post_pts(img_feats, [backbone_feats], img_inputs, bbox_pts, img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses_post_pts = self.post_pts_bbox_head.loss(*loss_inputs)
            losses.update(losses_post_pts)

        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        img_feats, _, __ = self.extract_img_feat(img, img_metas)
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        bbox_list = [dict(pts_bbox=bbox_pts[0])]

        return bbox_list
    
    # BEVInst Code Part
    def forward_post_pts(self, bev_feats, img_feats, img_inputs, bbox_pts, img_metas):
        priors = self.prepare_post_priors(bev_feats, bbox_pts)
        imgs, rt2imgs, post_rots, post_trans = \
            self.prepare_post_inputs(img_inputs)
        B, N, N_rgb, H, W = imgs[0].shape
        
        img_feats = self.prepare_4d_img_feats(img_feats)
        img_feats = self.post_img_neck(img_feats)
        if self.post_image_encoder is not None:
            img_feats = self.post_image_encoder(img_feats, (H, W))
        img_feats = [mlvl_feat.reshape(
            B, N, self.num_post_frame,
            mlvl_feat.shape[-3],
            mlvl_feat.shape[-2],
            mlvl_feat.shape[-1]) for mlvl_feat in img_feats]
        
        for i, img_meta in enumerate(img_metas):
            # img_meta['bda'] = bda[i]
            img_meta['rt2img'] = rt2imgs[i]
            img_meta['post_rot_xy_T'] = post_rots[i]
            img_meta['post_tran_xy'] = post_trans[i]
            img_meta['img_shape'] = (H, W)
            img_meta['out_size_factor'] = self.pts_bbox_head.test_cfg['out_size_factor']
            img_meta['voxel_size'] = self.pts_bbox_head.test_cfg['voxel_size']
        # img_metas[0]['imgs'] = imgs
        # img_metas[0]['pred'] = bbox_pts[0]['boxes_3d'].corners
        
        return self.post_pts_bbox_head(img_feats, img_metas, priors=priors, bev_feats=bev_feats)
    
    def prepare_post_priors(self, bev_feats, results):
        assert len(bev_feats) == 1
        bev_feat = bev_feats[0]
        H, W = bev_feat.shape[-2:]
        cfg = self.pts_bbox_head.test_cfg
        pc_range = self.post_pts_bbox_head.pc_range
        assert pc_range[:2] == cfg['pc_range']
        bboxes = []
        for result in results:
            bbox = result['boxes_3d'].tensor.clone().detach()
            bbox = normalize_bbox(bbox)
            if len(bbox) < self.num_proposal:
                N = self.num_proposal - len(bbox)
                bbox_padding_xyz = torch.rand(N, 3)
                bbox_padding_other = torch.Tensor([1, 1, 1, 1, 0, 0, 0])[None].repeat(N, 1) # l w h sin cos vx vy
                bbox_padding = torch.cat([bbox_padding_xyz[..., :2], 
                                        bbox_padding_other[..., :2], 
                                        bbox_padding_xyz[..., 2:3],
                                        bbox_padding_other[..., 2:]], dim=-1)
                bbox = torch.cat((bbox, bbox_padding), dim=0)
            else:
                _, indices = torch.topk(result['scores_3d'], self.num_proposal, sorted=False)
                bbox = bbox[indices]
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes).to(bev_feat.device)
        
        reference_points = bboxes[..., :2].clone()
        reference_points[..., 0] = (reference_points[..., 0] - cfg['pc_range'][0]
            ) / (cfg['out_size_factor'] * cfg['voxel_size'][0])
        reference_points[..., 1] = (reference_points[..., 1] - cfg['pc_range'][1]
            ) / (cfg['out_size_factor'] * cfg['voxel_size'][1])
        reference_points[..., 0] /= W
        reference_points[..., 1] /= H
        reference_points = (reference_points - 0.5) * 2

        if self.with_prior_grad:
            bev_feat = self.post_pts_bbox_head.prior_refiner(bev_feat)
        query_feat = F.grid_sample(bev_feat, reference_points.unsqueeze(1)).squeeze(2)
        query_feat = query_feat.permute(0, 2, 1)
        if self.deform_sample:
            query_feat = self.forward_deform_sample(query_feat, bev_feat, reference_points)
        priors = {'query_embed': query_feat, 'bboxes': bboxes}
        
        return priors
    
    def forward_deform_sample(self, query_feat, bev_feats, reference_points):
        bs, C, h,w = bev_feats.shape
        spatial_shapes = torch.tensor([[h, w]],dtype=torch.int64).to(query_feat.device)
        value=bev_feats.reshape(bs, C, h*w).permute(0,2,1)
        value = self.value_proj(value)
        _, num_value, _ = value.shape
        value = value.reshape(bs, num_value, self.num_heads, -1)
        query = query_feat
        bs,  num_query, embed_dims = query.shape
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points.unsqueeze(2)[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        query = self.dropout(output) + query_feat
        return query
    # def prepare_inputs(self, inputs):
    #     # split the inputs into each frame
    #     B, N, _, H, W = inputs[0].shape
    #     N = N // self.num_frame
    #     imgs = inputs[0].view(B, N, self.num_frame, 3, H, W)
    #     imgs = torch.split(imgs, 1, 2)
    #     imgs = [t.squeeze(2) for t in imgs]
    #     rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
    #     extra = [
    #         rots.view(B, self.num_frame, N, 3, 3),
    #         trans.view(B, self.num_frame, N, 3),
    #         intrins.view(B, self.num_frame, N, 3, 3),
    #         post_rots.view(B, self.num_frame, N, 3, 3),
    #         post_trans.view(B, self.num_frame, N, 3)
    #     ]
    #     extra = [torch.split(t, 1, 1) for t in extra]
    #     extra = [[p.squeeze(1) for p in t] for t in extra]
    #     rots, trans, intrins, post_rots, post_trans = extra
    #     return imgs, rots, trans, intrins, post_rots, post_trans, bda

    def prepare_post_inputs(self, inputs):
        B, N, _, H, W = inputs[0].shape
        imgs = inputs[0].view(B, N, 1, 3, H, W)
        imgs = torch.split(imgs, 1, dim=2)
        imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W
  
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]

        extra = [rots.view(B, 1, N, 3, 3),
                 trans.view(B, 1, N, 3),
                 intrins.view(B, 1, N, 3, 3),
                 post_rots.view(B, 1, N, 3, 3),
                 post_trans.view(B, 1, N, 3)]
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra] # each B x N x 3 (x 3)
        rots, trans, intrins, post_rots, post_trans = extra

        rt2imgs = []
        post_rots_T = []
        for t in range(self.num_post_frame):
            rots_t = rots[t].reshape(B*N, 3, 3)
            trans_t = trans[t].reshape(B*N, 3)
            intrins_t = intrins[t].reshape(B*N, 3, 3)
            post_rots_t = post_rots[t].reshape(B*N, 3, 3)
            rt2img = torch.stack([self.rt2img(
                rot, tran, intrin) for rot, tran, intrin in zip(
                    rots_t, trans_t, intrins_t)]).reshape(B, N, 4, 4)
            rt2imgs.append(rt2img)
            post_rot_T = torch.stack([
                post_rot[:2, :2].T for post_rot in post_rots_t
            ]).reshape(B, N, 2, 2)
            post_rots_T.append(post_rot_T)
        post_trans = post_trans[:self.num_post_frame]

        rt2imgs = torch.stack(rt2imgs, 2)
        post_rots_xy_T = torch.stack(post_rots_T, 2)
        post_trans_xy = torch.stack(post_trans, 2)[..., :2]
        
        return imgs, rt2imgs, post_rots_xy_T, post_trans_xy

    def rt2img(self, rot, tran, intrin):
        rt = torch.eye(4).to(rot.device)
        rt[:3, :3]= rot
        rt[:3, -1]= tran
        c2i = torch.eye(4).to(intrin.device)
        c2i[:3, :3] = intrin
        rt2i = rt.inverse().T @ c2i.T

        return rt2i
    
    def prepare_4d_img_feats(self, img_feats):
        assert len(img_feats) == self.num_post_frame
        assert len(img_feats[0]) == 4
        T = self.num_post_frame
        BN, C0, H0, W0 = img_feats[0][0].shape
        C1, H1, W1 = img_feats[0][1].shape[-3:]
        C2, H2, W2 = img_feats[0][2].shape[-3:]
        C3, H3, W3 = img_feats[0][3].shape[-3:]
        lvl0_feats_list = []
        lvl1_feats_list = []
        lvl2_feats_list = []
        lvl3_feats_list = []
        for pts_feat_lvl0, pts_feat_lvl1, \
            pts_feat_lvl2, pts_feat_lvl3 in img_feats:
            lvl0_feats_list.append(pts_feat_lvl0)
            lvl1_feats_list.append(pts_feat_lvl1)
            lvl2_feats_list.append(pts_feat_lvl2)
            lvl3_feats_list.append(pts_feat_lvl3)

        return (torch.stack(lvl0_feats_list, 1).reshape(BN*T, C0, H0, W0),
                torch.stack(lvl1_feats_list, 1).reshape(BN*T, C1, H1, W1),
                torch.stack(lvl2_feats_list, 1).reshape(BN*T, C2, H2, W2),
                torch.stack(lvl3_feats_list, 1).reshape(BN*T, C3, H3, W3))
    
    def simple_test_post_pts(self, bev_feats, img_feats, img, bbox_pts, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.forward_post_pts(bev_feats, [img_feats], img, bbox_pts, img_metas)
        bbox_list = self.post_pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_post_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        bev_feats, _, img_feats = self.extract_img_feat(img, img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(bev_feats, img_metas, rescale=rescale)
        bbox_post_pts = self.simple_test_post_pts(bev_feats, img_feats, img, bbox_pts, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_post_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            if self.post_test:
                return self.simple_post_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
            else:
                return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)
