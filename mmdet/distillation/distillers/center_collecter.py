import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER, build_distill_loss
from mmdet.models.roi_heads.standard_roi_head import build_roi_extractor
from collections import OrderedDict
from torch.nn.functional import adaptive_avg_pool2d,normalize


@DISTILLER.register_module()
class CenterCollerter(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """

    def __init__(self,
                 teacher1_cfg,
                 teacher2_cfg,
                 distill_cfg=None,
                 teacher1_pretrained=None,
                 teacher2_pretrained=None,
                 roi_extractor = None
                 ):

        super(CenterCollerter, self).__init__()
        print('start build feature collerctor -------------------------------------------------')
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.teacher1 = build_detector(teacher1_cfg.model,
                                      train_cfg=teacher1_cfg.get('train_cfg'),
                                      test_cfg=teacher1_cfg.get('test_cfg'))
        self.teacher2 = build_detector(teacher2_cfg.model,
                                      train_cfg=teacher2_cfg.get('train_cfg'),
                                      test_cfg=teacher2_cfg.get('test_cfg'))
        self.init_weights_teacher2(teacher1_pretrained,teacher2_pretrained)
        self.teacher1.eval()
        self.teacher2.eval()

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg
        print('start build roi_extractor -------------------------------------------------')
        self.easy_roi_extractor = build_roi_extractor(roi_extractor)
        print('build roi_extractor -------------------------------------------------')
        self.easy_Loss = torch.nn.L1Loss(reduction='mean')
        self.easy_conv = ConvModule(
            3,
            1,
            kernel_size=3,
            padding=1,
            inplace=False)

        teacher1_modules = dict(self.teacher1.named_modules())
        teacher2_modules = dict(self.teacher2.named_modules())


        def regitster_hooks(teacher1_module , teacher2_module):
            def hook_teacher1_forward(module, input, output):
                # print('in hook_teacher_forward------------------------------------',teacher_module,output.shape)
                self.register_buffer(teacher1_module, output)

            def hook_teacher2_forward(module, input, output):
                # print('in hook_teacher_forward------------------------------------',teacher_module,output.shape)
                self.register_buffer(teacher2_module, output)


            return hook_teacher1_forward,hook_teacher2_forward

        for item_loc in distill_cfg:


            teacher1_module = 'teacher1_' + item_loc.teacher1_module.replace('.', '_')
            teacher2_module = 'teacher2_' + item_loc.teacher2_module.replace('.', '_')

            print('-----------------------register----------------------------------', teacher1_module,teacher2_module)
            self.register_buffer(teacher1_module, None)
            self.register_buffer(teacher2_module, None)

            hook_teacher1_forward,hook_teacher2_forward= regitster_hooks(teacher1_module, teacher2_module)
            teacher1_modules[item_loc.teacher1_module].register_forward_hook(hook_teacher1_forward)
            teacher2_modules[item_loc.teacher2_module].register_forward_hook(hook_teacher2_forward)

        print('初始化完成 -------------------------------------------------D:\software_data\Py_project\year2023\KD_distiller\FPN_Twins_KD_收集聚类中心\mmdet\distillation\distillers\center_collecter.py')
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
            # feat = self.teacher1.extract_feat(img)
        easy_student_feature = self.easy_conv(img)
        easy_GT = torch.ones_like(easy_student_feature)
        student_loss = {}
        student_loss['easy_loss'] = self.easy_Loss(easy_student_feature,easy_GT)
        # student_loss = self.student.forward_train(img, img_metas, **kwargs)
        # print('interrupt  1--------------------get student loss')

        buffer_dict = dict(self.named_buffers())
        #
        # print('-----------------------------buffer_dict keys----------------------')
        # for key in buffer_dict.keys():
        #     print(key)

        for item_loc in self.distill_cfg:

            teacher1_module = 'teacher1_' + item_loc.teacher1_module.replace('.', '_')
            teacher2_module = 'teacher2_' + item_loc.teacher2_module.replace('.', '_')

            teacher1_feat = buffer_dict[teacher1_module]
            teacher2_feat = buffer_dict[teacher2_module]
            if 'backbone' in teacher1_module:

                self.teacher1_feature_size = teacher1_feat.size()
                self.teacher2_feature_size = teacher2_feat.size()

            # print('interrupt  2--------------------get registed buffer and send to loss')
            ########################################################
            gt_bboxes = kwargs['gt_bboxes']
            gt_labels = kwargs['gt_labels']
            gt_labels_add = None
            N, C, H, W = teacher1_feat.shape
            gt_bboxes_padding = []
            for i in range(N):
                # print('**' * 50)
                # print(gt_bboxes[i])
                # print('..'*50)
                # print(gt_labels[i])
                # print('--' * 50)
                # print(img_metas[i]['img_shape'])
                img_H = img_metas[i]['img_shape'][0]
                img_W = img_metas[i]['img_shape'][1]
                constandpadding = nn.ConstantPad2d(padding=(1,0,0,0),value=i)
                new_boxxes = torch.ones_like(gt_bboxes[i])
                new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * img_W
                new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * img_W
                new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * img_H
                new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * img_H

                new_boxxes_padding = constandpadding(new_boxxes)
                gt_bboxes_padding = new_boxxes_padding if i==0 else torch.cat((gt_bboxes_padding,new_boxxes_padding),dim=0)  #判断，如果是第一张图就直接赋值，第二张图就开始拼接
                gt_labels_add = gt_labels[i] if i==0 else torch.cat((gt_labels_add,gt_labels[i]),dim=0)
                ########################################################
                # print('mark----------------------------mmdet/distillation/distillers/center_collecter.py--------------------------------mark')
                # print(gt_bboxes[i].shape, gt_bboxes[i])
                # print(gt_labels[i].shape, gt_labels[i])
                # print(gt_bboxes_padding.shape, gt_bboxes_padding)
                # print(' ######################################################## teacher1_fpn')
                # print(type(teacher1_fpn),type(teacher2_fpn))
                # print(len(teacher1_fpn),len(teacher2_fpn))
                # print(type(teacher1_fpn[:4]),len(teacher1_fpn[:4]),teacher1_fpn[:4][0].shape)
                # print(' ######################################################## teacher2_fpn')
                # print(img_metas)
                #########################################################
            teacher1_bbox_feats = self.easy_roi_extractor(teacher1_fpn[:4], gt_bboxes_padding)
            teacher2_bbox_feats = self.easy_roi_extractor(teacher2_fpn[:4], gt_bboxes_padding)
            teacher1_feature_C = adaptive_avg_pool2d(teacher1_bbox_feats,(1,1))  #shape(Batch,256,1,1)
            teacher2_feature_C = adaptive_avg_pool2d(teacher2_bbox_feats,(1,1)) #shape(Batch,256,1,1)
            _,teacher1_feature_C_top20_index = torch.topk(teacher1_feature_C,20,dim=1,sorted=False)
            _,teacher2_feature_C_top20_index = torch.topk(teacher2_feature_C,20,dim=1,sorted=False)
            N,C,H,W = teacher1_bbox_feats.shape
            teacher1_bbox_feats_top20 = torch.zeros([N,20,H,W],dtype=teacher1_bbox_feats.dtype)
            teacher2_bbox_feats_top20 = torch.zeros_like(teacher1_bbox_feats_top20)
            for i in range(N):
                teacher1_bbox_feats_top20[i,:,:,:] = teacher1_bbox_feats[i,teacher1_feature_C_top20_index[i,:].squeeze(),:,:]
                teacher2_bbox_feats_top20[i,:,:,:] = teacher2_bbox_feats[i,teacher2_feature_C_top20_index[i,:].squeeze(),:,:]
                pass
            # print(teacher1_bbox_feats.shape,'----------------------teacher1_bbox_feats.shape')
            # print(teacher1_bbox_feats_top20.shape,'teacher1_bbox_feats_top20.shape,,,,,,,shape(batch,20,7,7)')
            # print(teacher1_feature_C_top20_index.squeeze().shape,'teacher1_feature_C_top20_index.squeeze().shape--------------------',teacher1_feature_C_top20_index.squeeze())
            teacher1_bbox_feats_top20_line = torch.flatten(torch.mean(teacher1_bbox_feats_top20,dim=1),1,-1)  #shape(batch,20,7,7)  -> (batch,49)
            teacher2_bbox_feats_top20_line = torch.flatten(torch.mean(teacher2_bbox_feats_top20,dim=1),1,-1)  #shape(batch,20,7,7)  -> (batch,49)
            # print(teacher1_bbox_feats_top20_line.shape,'teacher1_bbox_feats_top20_line------------#shape(batch,20,7,7)  -> (batch,49)')
            teacher1_bbox_feats_top20_line_normal = normalize(teacher1_bbox_feats_top20_line,p=2,dim=-1)
            teacher2_bbox_feats_top20_line_normal = normalize(teacher2_bbox_feats_top20_line,p=2,dim=-1)
            if teacher1_bbox_feats_top20_line_normal.shape[0] == gt_labels_add.shape[0]:
                for teacher1_object_line_feature,teacher2_object_line_feature,labels in zip(teacher1_bbox_feats_top20_line_normal.chunk(teacher1_bbox_feats_top20_line_normal.shape[0],dim=0) , teacher2_bbox_feats_top20_line_normal.chunk(teacher2_bbox_feats_top20_line_normal.shape[0],dim=0),gt_labels_add.chunk(gt_labels_add.shape[0],dim=0)):
                    teacher1_file = open('my_output/collected_center/teacher1/'+self.classes[labels[0]]+'.txt',mode='a')
                    teacher2_file = open('my_output/collected_center/teacher2/'+self.classes[labels[0]]+'.txt',mode='a')
                    teacher1_file.write(str(teacher1_object_line_feature.squeeze().tolist()) + ' \n')
                    teacher2_file.write(str(teacher2_object_line_feature.squeeze().tolist()) + ' \n')
                    teacher1_file.close()
                    teacher2_file.close()

                    pass


        return student_loss

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


