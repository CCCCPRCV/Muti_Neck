_base_ = [
    '../../fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py',
]
# model settings
find_unused_parameters=True
temp=0.8
alpha_fgd=0.0016
beta_fgd=0.0008
gamma_fgd=0.0008
lambda_fgd=0.000008
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = '/mnt/Data2/cjc/Download/Fire_fox/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.4.conv',
                         teacher_module = 'neck.fpn_convs.4.conv',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_4',
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
                    dict(student_module = 'neck.fpn_convs.3.conv',
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

student_cfg = 'configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py'
teacher_cfg = 'configs/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco.py'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
dataset_type = 'CocoDataset'
data_root = '/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/COCOFormatData/'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/CocoFormatData/annotations/instances_train.json',
        img_prefix='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/CocoFormatData/train/',
        ),
    val=dict(
        type='CocoDataset',
        ann_file='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/CocoFormatData/annotations/instances_val.json',
        img_prefix='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/CocoFormatData/val/',
        ),
    test=dict(
        type='CocoDataset',
        ann_file='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/CocoFormatData/annotations/instances_val.json',
        img_prefix='/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/CocoFormatData/val/',
        ))