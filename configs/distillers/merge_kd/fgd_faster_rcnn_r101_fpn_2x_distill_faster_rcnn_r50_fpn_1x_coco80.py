_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
temp=0.5
alpha_fgd=0.00005
beta_fgd=0.000025
gamma_fgd=0.00005
lambda_fgd=0.0000005
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = '/mnt/Data2/cjc/Download/Fire_fox/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth',
    init_student = False,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                   ]
    )

student_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
teacher_cfg = 'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=24)
dataset_type = 'CocoDataset'
data_root = '/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/COCOFormatData/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/CocoFormatData/annotations/instances_train.json',
        img_prefix='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/CocoFormatData/train/',
        ),
    val=dict(
        type='CocoDataset',
        ann_file='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/CocoFormatData/annotations/instances_val.json',
        img_prefix='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/CocoFormatData/val/',
        ),
    test=dict(
        type='CocoDataset',
        ann_file='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/CocoFormatData/annotations/instances_val.json',
        img_prefix='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_part/CocoFormatData/val/',
        ))