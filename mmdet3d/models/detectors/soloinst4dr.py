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
from .soloinst import SOLOInst
import copy
@DETECTORS.register_module()
class SOLOInst4dr(SOLOInst):
    def __init__(self, 
                 num_proposal_prev = 400, 
                 use_prev_embeds = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.history_priors = None
        self.use_prev_embeds = use_prev_embeds
        self.num_proposal_prev = num_proposal_prev
    # BEVInst Code Part
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
            img_feats_post = img_feats[0].clone().detach()
            bbox_pts = self.simple_test_pts([img_feats_post], img_metas, rescale=False)
            self.warp_history_priors(img_metas)
            outs = self.forward_post_pts([img_feats_post], [backbone_feats], img_inputs, bbox_pts, img_metas, use_prev_embeds = self.use_prev_embeds)
            self.update_history_priors(outs)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses_post_pts = self.post_pts_bbox_head.loss(*loss_inputs)
            losses.update(losses_post_pts)

        return losses
    
    def warp_history_priors(self, img_metas):
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas])

        if self.history_priors is None:
            return 

        curr_to_prev_lidar_rt = torch.stack([
            single_img_metas['curr_to_prev_lidar_rt']
            for single_img_metas in img_metas]).to(self.history_priors['bboxes'][0].device)
        
        bboxes_prev = self.history_priors['bboxes']
        embeds_prev = self.history_priors['embeds']
        results_new = list()
        embeds_new = list()
        for start, bbox_prev, embed_prev, curr_to_prev_lidar_rt in \
            zip(start_of_sequence, bboxes_prev, embeds_prev,curr_to_prev_lidar_rt):
            if start:
                results_new.append(torch.tensor([]))
                embeds_new.append(torch.tensor([]))
                continue

            bbox_prev_warp = self.priors_warp(bbox_prev, curr_to_prev_lidar_rt)
            results_new.append(bbox_prev_warp)
            embeds_new.append(embed_prev)
        self.history_priors = dict(bboxes = copy.copy(results_new),
                                   embeds = copy.copy(embeds_new))


    def update_history_priors(self, outs):
        bboxes_prev = outs['all_bbox_preds'][-1].clone().detach()
        scores_prev = outs['all_bbox_score'][-1].clone().detach()
        embeds_prev = torch.zeros_like(bboxes_prev)
        if self.use_prev_embeds:
            embeds_prev = outs['all_bbox_embeds'].clone().detach()
        results_bboxes = list()
        results_embeds = list()
        for bbox_prev, score_prev, embed_prev in zip(bboxes_prev, scores_prev, embeds_prev):
            score_prev = score_prev.squeeze(1)
            _, indices = torch.topk(score_prev, self.num_proposal_prev, sorted=False)
            bbox = bbox_prev[indices]
            embed = embed_prev[indices]
            results_bboxes.append(bbox)
            results_embeds.append(embed)
        self.history_priors = dict(bboxes = copy.copy(results_bboxes),
                                   embeds = copy.copy(results_embeds))
        # self.history_priors['bboxes'] = copy.copy(results_bboxes)
        # self.history_priors['embeds'] = copy.copy(results_embeds)
        
        
    def forward_post_pts(self, bev_feats, img_feats, img_inputs, bbox_pts, img_metas, use_prev_embeds=False):
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
        
        return self.post_pts_bbox_head(img_feats, img_metas, priors=priors, bev_feats=bev_feats, use_prev_embeds = use_prev_embeds)
    
    def priors_warp(self, bbox, l02l1):
        l12l0 = torch.inverse(l02l1)
        R = l12l0[:3,:3]
        whl = bbox[..., 3:6]
        xyz_prev = bbox[..., :3]
        xyz_prev = torch.cat((xyz_prev, torch.ones_like(xyz_prev[..., :1])), -1)
        xyz_warp = torch.matmul(l12l0, xyz_prev.permute(1,0)).permute(1,0)[...,:3]
        yaw_prev = bbox[..., 6:8]
        yaw_prev = torch.cat((yaw_prev, torch.zeros_like(yaw_prev[..., :1])), -1)
        yaw_warp = torch.matmul(R, yaw_prev.permute(1,0)).permute(1,0)[...,:2]
        v_prev = bbox[..., 8:]
        v_prev = torch.cat((v_prev, torch.zeros_like(v_prev[..., :1])), -1)
        v_warp = torch.matmul(R, v_prev.permute(1,0)).permute(1,0)[...,:2]
        bbox_warp = torch.cat((xyz_warp, whl, yaw_warp, v_warp), dim=-1)
        return bbox_warp
    
    def history_priors_fuse(self, results, img_metas):
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas])

        if self.history_priors is None:
            # self.history_priors = copy.copy(results)
            results_new = list()
            for result in results:
                bbox = result['boxes_3d'].tensor.clone().detach()
                bbox = normalize_bbox(bbox)
                priors_score = result['scores_3d']
                result_new = dict( boxes_3d = bbox, scores_3d=priors_score )
                results_new.append(result_new)
            self.history_priors = copy.copy(results_new)
            return results_new
        
        curr_to_prev_lidar_rt = torch.stack([
            single_img_metas['curr_to_prev_lidar_rt']
            for single_img_metas in img_metas]).to()
        
        fused_results = list()
        results_new = list()
        for start, result, history_prior, curr_to_prev_lidar_rt in \
            zip(start_of_sequence, results, self.history_priors, curr_to_prev_lidar_rt):
            if start:
                fused_results.append(result)
                continue

            bbox_prev = history_prior['boxes_3d']
            # bbox_prev = normalize_bbox(bbox_prev)
            bbox_prev_warp = self.priors_warp(bbox_prev, curr_to_prev_lidar_rt)
            # results['boxes_3d'] = bbox_prev
            history_priors_score = history_prior['scores_3d']
            bbox = result['boxes_3d'].tensor.clone().detach()
            bbox = normalize_bbox(bbox)
            priors_score = result['scores_3d']
            bboxes = torch.cat((bbox, bbox_prev_warp),0)
            scores = torch.cat((priors_score, history_priors_score),0)
            fused_result = dict( boxes_3d = bboxes, scores_3d=scores)
            fused_results.append(fused_result)

            result_new = dict( boxes_3d = bbox, scores_3d=priors_score )
            results_new.append(result_new)

        self.history_priors = copy.copy(results_new)
        return fused_results
    
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
        # bboxes_fuse = list()
        if not self.use_prev_embeds:
            priors_bboxes = None
            if self.history_priors is not None:
                priors_bboxes = self.history_priors['bboxes']

            for i in range(len(bboxes)):
                if priors_bboxes is not None and len(priors_bboxes[i]) > 0:
                    bboxes[i]=(torch.cat((bboxes[i], priors_bboxes[i].to(bboxes[i].device)), dim=0))
                else:
                    N = self.num_proposal_prev
                    bbox_padding_xyz = torch.rand(N, 3)
                    bbox_padding_other = torch.Tensor([1, 1, 1, 1, 0, 0, 0])[None].repeat(N, 1) # l w h sin cos vx vy
                    bbox_padding = torch.cat([bbox_padding_xyz[..., :2], 
                                            bbox_padding_other[..., :2], 
                                            bbox_padding_xyz[..., 2:3],
                                            bbox_padding_other[..., 2:]], dim=-1)
                    bboxes[i] = torch.cat((bboxes[i], bbox_padding), dim=0)

        if self.use_prev_embeds:
            priors_embeds = None
            if self.history_priors is not None:
                priors_embeds = self.history_priors['embeds']
            N = self.num_proposal_prev
            flag = False
            for i in range(len(bboxes)):
                if priors_embeds is None or priors_embeds is not None and len(priors_embeds[i]) == 0:
                    flag = True
                    break
            if flag:
                for i in range(len(bboxes)):
                    if priors_embeds is None or priors_embeds is not None and len(priors_embeds[i]) == 0:
                        bbox_padding_xyz = torch.rand(N, 3)
                        bbox_padding_other = torch.Tensor([1, 1, 1, 1, 0, 0, 0])[None].repeat(N, 1) # l w h sin cos vx vy
                        bbox_padding = torch.cat([bbox_padding_xyz[..., :2], 
                                                bbox_padding_other[..., :2], 
                                                bbox_padding_xyz[..., 2:3],
                                                bbox_padding_other[..., 2:]], dim=-1)
                        bboxes[i] = torch.cat((bboxes[i], bbox_padding), dim=0)

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
        priors = {'query_embed': query_feat, 'bboxes': bboxes}
        
        if self.use_prev_embeds and self.history_priors is not None:
            priors_bboxes = self.history_priors['bboxes']
            priors_embeds = self.history_priors['embeds']
            bboxes_fuse = list()
            embeds_fuse = list()
            if priors_bboxes is not None: 
                for i in range(len(priors_bboxes)):
                    if len(priors_embeds[i]) > 0:
                        bboxes_fuse.append(torch.cat((bboxes[i][:self.num_proposal_prev], priors_bboxes[i].to(bboxes[i].device)), dim=0))
                        embeds_fuse.append(torch.cat((query_feat[i][:self.num_proposal_prev], priors_embeds[i].to(bboxes[i].device)), dim=0))
                    else:
                        bboxes_fuse.append(bboxes[i])
                        embeds_fuse.append(query_feat[i])
                bboxes = torch.stack(bboxes_fuse).to(bev_feat.device)
                query_feat = torch.stack(embeds_fuse).to(bev_feat.device)
                priors = {'query_embed': query_feat, 'bboxes': bboxes}
        return priors

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
        outs = self.forward_post_pts(bev_feats, [img_feats], img, bbox_pts, img_metas, use_prev_embeds = self.use_prev_embeds)
        self.update_history_priors(outs)
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
        self.warp_history_priors(img_metas)
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
