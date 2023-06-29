import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER, build_distill_loss
from collections import OrderedDict
import copy
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from torch.nn.functional import adaptive_avg_pool2d,normalize,mse_loss
from mmdet.models.roi_heads.standard_roi_head import build_roi_extractor


@DISTILLER.register_module()
class MergeEDDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """

    def __init__(self,
                 teacher1_cfg,
                 teacher2_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher1_pretrained=None,
                 teacher2_pretrained=None,
                 student_pretrain=None,
                 defualt_init_teacher='teacher1',
                 init_student=False):

        super(MergeEDDistiller, self).__init__()
        self.teacher1 = build_detector(teacher1_cfg.model,
                                      train_cfg=teacher1_cfg.get('train_cfg'),
                                      test_cfg=teacher1_cfg.get('test_cfg'))
        self.teacher2 = build_detector(teacher2_cfg.model,
                                      train_cfg=teacher2_cfg.get('train_cfg'),
                                      test_cfg=teacher2_cfg.get('test_cfg'))
        self.init_weights_teacher2(teacher1_pretrained,teacher2_pretrained)
        self.teacher1.eval()
        self.teacher2.eval()
        ########################################################################  新加的代码
        # def get_centers_from_txt(file_path = 'my_output/collected_center/teacher1_centers.txt'):
        #     centers_dict = {}
        #     file_hander = open(file_path,mode='r')
        #     for line in file_hander.readlines():
        #         center,class_index = line.strip().split('|')
        #         centers_dict[class_index.strip()] = torch.tensor(eval(center))
        #     return centers_dict
        # self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))
        # self.teacher1_centers = get_centers_from_txt(teacher1_center_txt)
        # self.teacher2_centers = get_centers_from_txt(teacher2_center_txt)
        # roi_extractor = dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32])
        # self.easy_roi_extractor = build_roi_extractor(roi_extractor)
        #######################################################################

        self.student = build_detector(student_cfg.model,
                                      train_cfg=student_cfg.get('train_cfg'),
                                      test_cfg=student_cfg.get('test_cfg'))
        if init_student:
            if student_pretrain == None:
                if defualt_init_teacher =='teacher1':
                    t_checkpoint = _load_checkpoint(teacher1_pretrained)
                else:
                    t_checkpoint = _load_checkpoint(teacher2_pretrained)
                all_name = []
                for name, v in t_checkpoint["state_dict"].items():
                    if name.startswith("backbone."):
                        continue
                    else:
                        all_name.append((name, v))

                state_dict = OrderedDict(all_name)
                load_state_dict(self.student, state_dict)
            else:
                load_checkpoint(self.student, student_pretrain)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher1_modules = dict(self.teacher1.named_modules())
        teacher2_modules = dict(self.teacher2.named_modules())


        def regitster_hooks(student_module1,student_module2, teacher1_module , teacher2_module):
            def hook_teacher1_forward(module, input, output):
                # print('in hook_teacher_forward------------------------------------',teacher_module,output.shape)
                self.register_buffer(teacher1_module, output)

            def hook_teacher2_forward(module, input, output):
                # print('in hook_teacher_forward------------------------------------',teacher_module,output.shape)
                self.register_buffer(teacher2_module, output)

            def hook_student1_forward(module, input, output):
                # print('in hook_student_forward------------------------------------', student_module,output.shape)
                self.register_buffer(student_module1, output)

            def hook_student2_forward(module, input, output):
                # print('in hook_student_forward------------------------------------', student_module,output.shape)
                self.register_buffer(student_module2, output)

            return hook_teacher1_forward,hook_teacher2_forward, hook_student1_forward,hook_student2_forward

        for item_loc in distill_cfg:

            student_module1 = 'student_' + item_loc.student_module1.replace('.', '_')
            student_module2 = 'student_' + item_loc.student_module2.replace('.', '_')
            teacher1_module = 'teacher1_' + item_loc.teacher1_module.replace('.', '_')
            teacher2_module = 'teacher2_' + item_loc.teacher2_module.replace('.', '_')

            print('-----------------------register----------------------------------', student_module1, teacher1_module)
            self.register_buffer(student_module1, None)
            self.register_buffer(student_module2, None)
            self.register_buffer(teacher1_module, None)
            self.register_buffer(teacher2_module, None)

            hook_teacher1_forward,hook_teacher2_forward, hook_student1_forward,hook_student2_forward = regitster_hooks(student_module1,student_module2, teacher1_module, teacher2_module)
            teacher1_modules[item_loc.teacher1_module].register_forward_hook(hook_teacher1_forward)
            teacher2_modules[item_loc.teacher2_module].register_forward_hook(hook_teacher2_forward)
            student_modules[item_loc.student_module1].register_forward_hook(hook_student1_forward)
            student_modules[item_loc.student_module2].register_forward_hook(hook_student2_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)



    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def init_weights_teacher2(self, teacherpath1=None , teacherpath2 = None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        checkpoint1 = load_checkpoint(self.teacher1, filename=teacherpath1, map_location='cpu')
        checkpoint2 = load_checkpoint(self.teacher2, filename=teacherpath2, map_location='cpu')


    def forward_train(self, img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        with torch.no_grad():
            self.teacher1.eval()
            self.teacher2.eval()
            teacher1_fpn = self.teacher1.extract_feat(img)
            teacher2_fpn = self.teacher2.extract_feat(img)
            teacher1_result = self.teacher1.simple_test(img,img_metas)
            teacher2_result = self.teacher2.simple_test(img,img_metas)
            # feat = self.teacher1.extract_feat(img)
        ##################################################################################################################################################################
        # print('mmdet/distillation/distillers/merge_distiller_add_proposal.py---------------------------------------------------------------forward')
        # # print(teacher1_result)
        # # print('teacher1_result-----------------------------------------------------------------------------------------------------------')
        # # print(teacher1_result[0])
        # print('teacher1_result-----------------------------------------------------------------------------------------------------------')
        # print(teacher1_result[0][0])
        # print('teacher1_result-----------------------------------------------------------------------------------------------------------')
        # # print(teacher2_result)
        # # print('teacher2_result-----------------------------------------------------------------------------------------------------------')
        # # print(teacher2_result[0])
        # print('teacher2_result-----------------------------------------------------------------------------------------------------------')
        # print(teacher2_result[0][0][0])
        # print('mmdet/distillation/distillers/merge_distiller_add_proposal.py---------------------------------------------------------------forward')
        ###################################################################################
        # '''计算两个教师对应的边界区域mask，中心点距离mask'''
        # assert teacher1_fpn.shape[0] == len(teacher1_result)
        # gt_bboxes = kwargs['gt_bboxes']
        # gt_labels = kwargs['gt_labels']
        # new_scaled_gt_bboxes = []
        # gt_bboxes_padding = []
        # N,C,H,W = teacher1_fpn.shape
        # teacher1_edge_mask = torch.ones([N,H,W],dtype=teacher1_fpn.dtype)*4
        # teacher2_edge_mask = torch.ones_like(teacher1_edge_mask)*4
        # teacher1_distance_mask = torch.ones_like(teacher1_edge_mask)
        # teacher2_distance_mask = torch.ones_like(teacher1_edge_mask)
        # ###########       4个重要的mask，分别是教师1和教师2的proposal叠加mask和离中心点距离的mask
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = torch.floor(gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W)
        #     new_boxxes[:, 2] = torch.ceil(gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W)
        #     new_boxxes[:, 1] = torch.floor(gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H)
        #     new_boxxes[:, 3] = torch.ceil(gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H)
        #     new_scaled_gt_bboxes.append(new_boxxes)
        #     constandpadding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=i) #加标签，其中value表示这个截出来的roi是属于哪张图片的
        #     new_boxxes_padding = constandpadding(new_boxxes)
        #     gt_bboxes_padding = new_boxxes_padding if i == 0 else torch.cat((gt_bboxes_padding, new_boxxes_padding),dim=0)  # 判断，如果是第一张图就直接赋值，第二张图就开始拼接
        #     gt_labels_add = gt_labels[i] if i == 0 else torch.cat((gt_labels_add, gt_labels[i]), dim=0)
        #     for j in range(len(teacher1_result[i])):
        #         ####teacher1
        #         if teacher1_result[i][j].shape[0] != 0:
        #             teacher1_new_class_propsal_bboxes = teacher1_result[i][j]
        #             teacher1_new_class_propsal_bboxes[:,0] = np.floor(teacher1_new_class_propsal_bboxes / img_metas[i]['img_shape'][1] * W)
        #             teacher1_new_class_propsal_bboxes[:,2] = np.ceil(teacher1_new_class_propsal_bboxes / img_metas[i]['img_shape'][1] * W)
        #             teacher1_new_class_propsal_bboxes[:,1] = np.floor(teacher1_new_class_propsal_bboxes / img_metas[i]['img_shape'][0] * H)
        #             teacher1_new_class_propsal_bboxes[:,3] = np.ceil(teacher1_new_class_propsal_bboxes / img_metas[i]['img_shape'][0] * H)
        #             for k in range(teacher1_result[i][j].shape[0]):
        #                 if teacher1_new_class_propsal_bboxes[k,4]>0.2:
        #                     teacher1_edge_mask[i,teacher1_new_class_propsal_bboxes[k,1]:teacher1_new_class_propsal_bboxes[k,3],teacher1_new_class_propsal_bboxes[k,0]:teacher1_new_class_propsal_bboxes[k,2]] += teacher1_new_class_propsal_bboxes[k,4]*2  #权重赋值，赋值方式为叠加，越是重合的地方权重越大
        #                 else:
        #                     break
        #                 pass
        #             pass
        #         else:
        #             pass
        #         ####teacher2
        #         if teacher2_result[i][j].shape[0] != 0:
        #             teacher2_new_class_propsal_bboxes = teacher1_result[i][j]
        #             teacher2_new_class_propsal_bboxes[:,0] = np.floor(teacher2_new_class_propsal_bboxes / img_metas[i]['img_shape'][1] * W)
        #             teacher2_new_class_propsal_bboxes[:,2] = np.ceil(teacher2_new_class_propsal_bboxes / img_metas[i]['img_shape'][1] * W)
        #             teacher2_new_class_propsal_bboxes[:,1] = np.floor(teacher2_new_class_propsal_bboxes / img_metas[i]['img_shape'][0] * H)
        #             teacher2_new_class_propsal_bboxes[:,3] = np.ceil(teacher2_new_class_propsal_bboxes / img_metas[i]['img_shape'][0] * H)
        #             for k in range(teacher1_result[i][j].shape[0]):
        #                 if teacher2_new_class_propsal_bboxes[k,4]>0.2:
        #                     teacher2_edge_mask[i,teacher2_new_class_propsal_bboxes[k,1]:teacher2_new_class_propsal_bboxes[k,3],teacher2_new_class_propsal_bboxes[k,0]:teacher2_new_class_propsal_bboxes[k,2]] += teacher2_new_class_propsal_bboxes[k,4]*2
        #                 else:
        #                     break
        #                 pass
        #             pass
        #         else:
        #             pass
        #
        #         pass
        #     pass
        # teacher1_bbox_feats = self.easy_roi_extractor(teacher1_fpn[:4], gt_bboxes_padding)
        # teacher2_bbox_feats = self.easy_roi_extractor(teacher2_fpn[:4], gt_bboxes_padding)
        # teacher1_feature_C = adaptive_avg_pool2d(teacher1_bbox_feats, (1, 1))  # shape(Batch,256,1,1)
        # teacher2_feature_C = adaptive_avg_pool2d(teacher2_bbox_feats, (1, 1))  # shape(Batch,256,1,1)
        # _, teacher1_feature_C_top20_index = torch.topk(teacher1_feature_C, 20, dim=1, sorted=False)
        # _, teacher2_feature_C_top20_index = torch.topk(teacher2_feature_C, 20, dim=1, sorted=False)
        # N_roi, C_roi, H_roi, W_roi = teacher1_bbox_feats.shape
        # teacher1_bbox_feats_top20 = torch.zeros([N_roi, 20, H_roi, W_roi], dtype=teacher1_bbox_feats.dtype)
        # teacher2_bbox_feats_top20 = torch.zeros_like(teacher1_bbox_feats_top20)
        # for i in range(N):
        #     teacher1_bbox_feats_top20[i, :, :, :] = teacher1_bbox_feats[i,teacher1_feature_C_top20_index[i, :].squeeze(), :, :]
        #     teacher2_bbox_feats_top20[i, :, :, :] = teacher2_bbox_feats[i,teacher2_feature_C_top20_index[i, :].squeeze(), :, :]
        #     pass
        # # print(teacher1_bbox_feats.shape,'----------------------teacher1_bbox_feats.shape')
        # # print(teacher1_bbox_feats_top20.shape,'teacher1_bbox_feats_top20.shape,,,,,,,shape(batch,20,7,7)')
        # # print(teacher1_feature_C_top20_index.squeeze().shape,'teacher1_feature_C_top20_index.squeeze().shape--------------------',teacher1_feature_C_top20_index.squeeze())
        # teacher1_bbox_feats_top20_line = torch.flatten(torch.mean(teacher1_bbox_feats_top20, dim=1), 1,-1)  # shape(batch,20,7,7)  -> (batch,49)
        # teacher2_bbox_feats_top20_line = torch.flatten(torch.mean(teacher2_bbox_feats_top20, dim=1), 1,-1)  # shape(batch,20,7,7)  -> (batch,49)
        # # print(teacher1_bbox_feats_top20_line.shape,'teacher1_bbox_feats_top20_line------------#shape(batch,20,7,7)  -> (batch,49)')
        # teacher1_bbox_feats_top20_line_normal = normalize(teacher1_bbox_feats_top20_line, p=2, dim=-1)
        # teacher2_bbox_feats_top20_line_normal = normalize(teacher2_bbox_feats_top20_line, p=2, dim=-1)
        # index_count = 0
        # for i in range(N):
        #     for j in range(new_scaled_gt_bboxes[i].shape[0]):
        #         object_class = gt_labels[i][j]
        #         teacher1_distance = mse_loss(teacher1_bbox_feats_top20_line_normal[index_count,:],self.teacher1_centers[str(object_class)])
        #         teacher2_distance = mse_loss(teacher2_bbox_feats_top20_line_normal[index_count,:],self.teacher1_centers[str(object_class)])
        #         teacher1_distance_mask[i,new_scaled_gt_bboxes[i][j,1]:new_scaled_gt_bboxes[i][j,3]] = torch.where(teacher1_distance_mask[i,new_scaled_gt_bboxes[i][j,1]:new_scaled_gt_bboxes[i][j,3]]<1/teacher1_distance,teacher1_distance_mask[i,new_scaled_gt_bboxes[i][j,1]:new_scaled_gt_bboxes[i][j,3]],1/teacher1_distance)
        #         teacher2_distance_mask[i,new_scaled_gt_bboxes[i][j,1]:new_scaled_gt_bboxes[i][j,3]] = torch.where(teacher2_distance_mask[i,new_scaled_gt_bboxes[i][j,1]:new_scaled_gt_bboxes[i][j,3]]<1/teacher1_distance,teacher2_distance_mask[i,new_scaled_gt_bboxes[i][j,1]:new_scaled_gt_bboxes[i][j,3]],1/teacher2_distance)
        #         index_count += 1
        #         pass
        #     pass
        #####################################################################################################################################################################

        student_loss = self.student.forward_train(img, img_metas, **kwargs)
        # print('interrupt  1--------------------get student loss')

        buffer_dict = dict(self.named_buffers())
        #
        # print('-----------------------------buffer_dict keys----------------------')
        # for key in buffer_dict.keys():
        #     print(key)
        #
        for item_loc in self.distill_cfg:

            student_module1 = 'student_' + item_loc.student_module1.replace('.', '_')
            student_module2 = 'student_' + item_loc.student_module2.replace('.', '_')
            teacher1_module = 'teacher1_' + item_loc.teacher1_module.replace('.', '_')
            teacher2_module = 'teacher2_' + item_loc.teacher2_module.replace('.', '_')

            student_feat1 = buffer_dict[student_module1]
            student_feat2 = buffer_dict[student_module2]
            teacher1_feat = buffer_dict[teacher1_module]
            teacher2_feat = buffer_dict[teacher2_module]
            if 'backbone' in student_module1 and 'backbone' in teacher1_module:
                self.student_feature_size = student_feat1.size()
                self.teacher1_feature_size = teacher1_feat.size()
                self.teacher2_feature_size = teacher2_feat.size()

            # print('interrupt  2--------------------get registed buffer and send to loss')
            for item_loss in item_loc.methods:
                loss_name = item_loss.name

                if 'encoder' in loss_name:
                    student_loss[loss_name] = self.distill_losses[loss_name](student_feat1, teacher1_feat,
                                                                             kwargs['gt_bboxes'], img_metas,
                                                                             self.student_feature_size)
                    # print('interrupt  3--------------------get loss of tansformer encoder    encoder')
                else:
                    teacher1_fpn_clone = copy.deepcopy(teacher1_fpn)
                    teacher2_fpn_clone = copy.deepcopy(teacher2_fpn)
                    teacher1_result_clone = copy.deepcopy(teacher1_result)
                    teacher2_result_clone = copy.deepcopy(teacher2_result)

                    student_loss[loss_name],student_loss[loss_name+'object'] = self.distill_losses[loss_name](student_feat1,student_feat2, teacher1_feat,teacher2_feat,
                                                                             kwargs['gt_bboxes'], kwargs['gt_labels'],teacher1_fpn_clone,teacher2_fpn_clone,teacher1_result_clone,teacher2_result_clone,
                                                                             img_metas)
                    # student_loss[loss_name] = self.distill_losses[loss_name](preds_S1=student_feat1, preds_S2=student_feat2,
                    #                                                          preds_T1=teacher1_feat, preds_T2=teacher2_feat,
                    #                                                          gt_bboxes=kwargs['gt_bboxes'], gt_labels=kwargs['gt_labels'],
                    #                                                          teacher1_fpn=teacher1_fpn, teacher2_fpn=teacher2_fpn,
                    #                                                          teacher1_result=teacher1_result, teacher2_result=teacher2_result,
                    #                                                          img_metas=img_metas)

                    # print('interrupt  4--------------------get loss of tansformer decoder  or  backbone')

        return student_loss

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


