import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, ConvModule, bias_init_with_prob
from mmcv.runner import force_fp32
                        
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.util import normalize_bbox


@HEADS.register_module()
class BEVInstHead(DETRHead):
    def __init__(self,
                 *args,
                 with_prior_grad=False,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 random_bbox_xy=True,
                 fp_16=False,
                 **kwargs):
        self.with_prior_grad = with_prior_grad
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.fp_16 = fp_16
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        self.random_bbox_xy = random_bbox_xy
        super(BEVInstHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if self.with_prior_grad:
            # self.prior_refiner = nn.Linear(self.embed_dims, self.embed_dims)
            self.prior_refiner = ConvModule(
                self.embed_dims,
                self.embed_dims,
                kernel_size=3,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                bias='auto',)

        if not self.as_two_stage:
            if self.num_query > self.transformer.num_proposal:
                num_query_rand = self.num_query - self.transformer.num_proposal
                self.query_embedding = nn.Embedding(
                    num_query_rand, self.embed_dims)
                    # self.embed_dims * 2)
                self.bbox_rand_xyz = nn.Embedding(num_query_rand, 3)
                # ori
                if self.random_bbox_xy:
                    self.bbox_rand_xyz.weight.data[:, :2].uniform_(0,1)
                    self.bbox_rand_xyz.weight.data = inverse_sigmoid(self.bbox_rand_xyz.weight.data)
                    self.bbox_rand_xyz.weight.data[:, :2].requires_grad = False
                
                #  fp16_modify
                if self.fp_16:
                    assert not self.random_bbox_xy, "fp16 cannot support random_bbox_xy"
                    nn.init.constant_(self.bbox_rand_xyz.weight.data[:,:3],0)
                    # nn.init.xavier_uniform_(self.bbox_rand_xyz.weight.data[:,:3])
                    # self.bbox_rand_xyz.weight.data[:, 2].uniform_(0,1)
                    # self.bbox_rand_xyz.weight.data = inverse_sigmoid(self.bbox_rand_xyz.weight.data,1e-8)
                    self.bbox_rand_xyz.weight.data[:, :3].requires_grad = False


    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, img_metas, only_query=False, priors=None, **kwargs):    
        if self.num_query > self.transformer.num_proposal:
            num_query_rand = self.num_query - self.transformer.num_proposal
            query_rand = self.query_embedding.weight
            bbox_rand_xyz = self.bbox_rand_xyz.weight
            bbox_rand_other = torch.Tensor([1, 1, 1, 1, 0, 0, 0])[None].repeat(num_query_rand, 1).to(bbox_rand_xyz.device) # l w h sin cos vx vy
            bbox_rand = torch.cat([bbox_rand_xyz[..., :2], 
                                    bbox_rand_other[..., :2], 
                                    bbox_rand_xyz[..., 2:3],
                                    bbox_rand_other[..., 2:]], dim=-1)
            query_embeds = {'query_embed': query_rand, 'bboxes': bbox_rand}
        else:
            query_embeds = None

        if priors is not None:
            bboxes = priors['bboxes']
            bboxes[..., 0:1] = inverse_sigmoid((bboxes[...,0:1]-self.pc_range[0])/(self.pc_range[3] - self.pc_range[0]))
            bboxes[..., 1:2] = inverse_sigmoid((bboxes[...,1:2]-self.pc_range[1])/(self.pc_range[4] - self.pc_range[1]))
            bboxes[..., 4:5] = inverse_sigmoid((bboxes[...,4:5]-self.pc_range[2])/(self.pc_range[5] - self.pc_range[2]))
            priors['bboxes'] = bboxes
        else:
            bboxes = None
        
        hs, inter_bboxes = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            img_metas=img_metas,
            priors=priors,
            **kwargs)
        hs = hs.permute(0, 2, 1, 3)

        if only_query:
            return kwargs['prev_info']

        inter_bboxes[..., 0:1] = inter_bboxes[...,0:1].sigmoid()*(self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        inter_bboxes[..., 1:2] = inter_bboxes[...,1:2].sigmoid()*(self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        inter_bboxes[..., 4:5] = inter_bboxes[...,4:5].sigmoid()*(self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        outputs_classes = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.cls_branches[lvl](hs[lvl])
            outputs_classes.append(outputs_class)
        outputs_classes = torch.stack(outputs_classes)
        
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': inter_bboxes,
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        try:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        except:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        try:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        except:
            assert sampling_result.pos_gt_bboxes.numel() == 0
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes.view(-1, bbox_targets.shape[-1])
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list