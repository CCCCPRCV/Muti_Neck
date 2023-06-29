# _base_ = [
#     '../../_base_/datasets/coco_detection.py',
#     '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
# ]
_base_ = [
    '../../fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py'
]
#dataset setting


# model settings



find_unused_parameters=True
temp=0.8
alpha_fgd=0.001
beta_fgd=0.0005
gamma_fgd=0.0005
lambda_fgd=0.000005
distiller = dict(
    type='MergeDistiller',
    teacher1_pretrained = '/mnt/Data2/cjc/Download/Fire_fox/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth',
    teacher2_pretrained = '/mnt/Data2/cjc/Download/Fire_fox/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coaco-511424d6.pth',
    init_student = False,
    student_pretrain = '/mnt/Data2/cjc/Download/Fire_fox/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth',
    defualt_init_teacher = 'teacher2',
    distill_cfg = [
                    # dict(student_module = 'backbone.layer4',
                    #      teacher_module = 'backbone.layer4',
                    #      output_hook = True,
                    #      methods=[dict(type='FeatureLoss_small',
                    #                    name='resnet_feature_loss',
                    #                    student_channels = 512,
                    #                    teacher_channels = 2048,
                    #                    temp = temp,
                    #                    alpha_fgd=alpha_fgd,
                    #                    beta_fgd=beta_fgd,
                    #                    gamma_fgd=gamma_fgd,
                    #                    lambda_fgd=lambda_fgd,
                    #                    )
                    #             ]
                    #     ),
                    # dict(student_module = 'bbox_head.transformer.encoder.layers.0',
                    #      teacher_module = 'bbox_head.transformer.encoder.layers.0',
                    #      output_hook = True,
                    #      methods=[dict(type='EncoderFeatureLoss',
                    #                    name='transformer_encoderlayer0_loss',
                    #                    student_channels = 256,
                    #                    teacher_channels = 256,
                    #                     temp = temp,
                    #                    )
                    #             ]
                    #     ),
                    dict(student_module1='neck.fpn_convs1.4.conv',
                         student_module2='neck.fpn_convs2.4.conv',
                         teacher1_module='neck.fpn_convs.4.conv',
                         teacher2_module='neck.fpn_convs.4.conv',
                         output_hook=True,
                         methods=[dict(type='MergeLoss',
                                       name='loss_fgd_fpn_4',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                  ]
                         ),
                    dict(student_module1='neck.fpn_convs1.3.conv',
                         student_module2='neck.fpn_convs2.3.conv',
                         teacher1_module='neck.fpn_convs.3.conv',
                         teacher2_module='neck.fpn_convs.3.conv',
                         output_hook=True,
                         methods=[dict(type='MergeLoss',
                                       name='loss_fgd_fpn_3',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                  ]
                         ),
                    dict(student_module1='neck.fpn_convs1.2.conv',
                         student_module2='neck.fpn_convs2.2.conv',
                         teacher1_module='neck.fpn_convs.2.conv',
                         teacher2_module='neck.fpn_convs.2.conv',
                         output_hook=True,
                         methods=[dict(type='MergeLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                  ]
                         ),
                    dict(student_module1='neck.fpn_convs1.1.conv',
                         student_module2='neck.fpn_convs2.1.conv',
                         teacher1_module='neck.fpn_convs.1.conv',
                         teacher2_module='neck.fpn_convs.1.conv',
                         output_hook=True,
                         methods=[dict(type='MergeLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                  ]
                         ),
                        dict(student_module1='neck.fpn_convs1.0.conv',
                             student_module2='neck.fpn_convs2.0.conv',
                             teacher1_module='neck.fpn_convs.0.conv',
                             teacher2_module='neck.fpn_convs.0.conv',
                             output_hook=True,
                             methods=[dict(type='MergeLoss',
                                           name='loss_fgd_fpn_0',
                                           student_channels=256,
                                           teacher_channels=256,
                                           temp=temp,
                                           alpha_fgd=alpha_fgd,
                                           beta_fgd=beta_fgd,
                                           gamma_fgd=gamma_fgd,
                                           lambda_fgd=lambda_fgd,
                                           )
                                      ]
                             )

    ]
    )

student_cfg = 'configs/distillers/merge_kd/new_config/fcos_caffe_fpn_twins_gn_head.py'
teacher1_cfg = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
teacher2_cfg = 'configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
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
