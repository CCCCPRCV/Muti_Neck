import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER, build_distill_loss
from collections import OrderedDict
from mmdet.core.bbox.iou_calculators import build_iou_calculator


@DISTILLER.register_module()
class MergeDistiller(BaseDetector):
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

        super(MergeDistiller, self).__init__()

        self.teacher1 = build_detector(teacher1_cfg.model,
                                      train_cfg=teacher1_cfg.get('train_cfg'),
                                      test_cfg=teacher1_cfg.get('test_cfg'))
        self.teacher2 = build_detector(teacher2_cfg.model,
                                      train_cfg=teacher2_cfg.get('train_cfg'),
                                      test_cfg=teacher2_cfg.get('test_cfg'))
        self.init_weights_teacher2(teacher1_pretrained,teacher2_pretrained)
        self.teacher1.eval()
        self.teacher2.eval()
        self.iou_calculator = build_iou_calculator(dict(type='BboxOverlaps2D'))

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
            # teacher1_result = self.teacher1.forward_train(img, img_metas, **kwargs)
            # teacher2_result = self.teacher2.forward_train(img, img_metas, **kwargs)
            teacher1_result = self.teacher1.extract_feat(img)
            teacher2_result = self.teacher2.extract_feat(img)
            # feat = self.teacher1.extract_feat(img)

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
                    student_loss[loss_name] = self.distill_losses[loss_name](student_feat1,student_feat2, teacher1_feat,teacher2_feat,
                                                                             kwargs['gt_bboxes'], kwargs['gt_labels'],
                                                                             img_metas)
                    # print('interrupt  4--------------------get loss of tansformer decoder  or  backbone')

        return student_loss

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


