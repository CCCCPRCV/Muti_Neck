# _base_ = [
#     '../../_base_/datasets/coco_detection.py',
#     '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
# ]
_base_ = [
    '../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]
#dataset setting


# model settings



find_unused_parameters=True
temp=0.8
alpha_fgd=0.001
beta_fgd=0.0005
gamma_fgd=0.0005
lambda_fgd=0.000005
teacher1_similarity = 0.5
distiller = dict(
    type='MergeEDDistiller',
    teacher1_pretrained = 'my_output/my_pth/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth',
    teacher2_pretrained = 'my_output/my_pth/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth',
    init_student = False,
    student_pretrain = 'my_output/my_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
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
                    # dict(student_module1='neck.fpn_convs1.4.conv',
                    #      student_module2='neck.fpn_convs2.4.conv',
                    #      teacher1_module='neck.fpn_convs.4.conv',
                    #      teacher2_module='neck.fpn_convs.4.conv',
                    #      output_hook=True,
                    #      methods=[dict(type='MergeLoss',
                    #                    name='loss_fgd_fpn_4',
                    #                    student_channels=256,
                    #                    teacher_channels=256,
                    #                    temp=temp,
                    #                    alpha_fgd=alpha_fgd,
                    #                    beta_fgd=beta_fgd,
                    #                    gamma_fgd=gamma_fgd,
                    #                    lambda_fgd=lambda_fgd,
                    #                    )
                    #               ]
                    #      ),
                    dict(student_module1='neck.fpn_convs1.3.conv',
                         student_module2='neck.fpn_convs2.3.conv',
                         teacher1_module='neck.fpn_convs.3.conv',
                         teacher2_module='neck.fpn_convs.3.conv',
                         output_hook=True,
                         methods=[dict(type='MergeEDLoss',
                                       name='loss_fgd_fpn_3',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       teacher1_center_txt = 'my_output/collected_center/teacher1_centers.txt',
                                       teacher2_center_txt = 'my_output/collected_center/teacher2_centers.txt',
                                       roi_extractor=dict(
                                                         type='SingleRoIExtractor',
                                                         roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                                         out_channels=256,
                                                         featmap_strides=[4, 8, 16, 32])
                                                        )
                                  ]
                         ),
                    dict(student_module1='neck.fpn_convs1.2.conv',
                         student_module2='neck.fpn_convs2.2.conv',
                         teacher1_module='neck.fpn_convs.2.conv',
                         teacher2_module='neck.fpn_convs.2.conv',
                         output_hook=True,
                         methods=[dict(type='MergeEDLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       teacher1_center_txt='my_output/collected_center/teacher1_centers.txt',
                                       teacher2_center_txt='my_output/collected_center/teacher2_centers.txt',
                                       roi_extractor=dict(
                                           type='SingleRoIExtractor',
                                           roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                           out_channels=256,
                                           featmap_strides=[4, 8, 16, 32])
                                       )
                                  ]
                         ),
                    dict(student_module1='neck.fpn_convs1.1.conv',
                         student_module2='neck.fpn_convs2.1.conv',
                         teacher1_module='neck.fpn_convs.1.conv',
                         teacher2_module='neck.fpn_convs.1.conv',
                         output_hook=True,
                         methods=[dict(type='MergeEDLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels=256,
                                       teacher_channels=256,
                                       temp=temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       teacher1_center_txt='my_output/collected_center/teacher1_centers.txt',
                                       teacher2_center_txt='my_output/collected_center/teacher2_centers.txt',
                                       roi_extractor=dict(
                                           type='SingleRoIExtractor',
                                           roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                           out_channels=256,
                                           featmap_strides=[4, 8, 16, 32])
                                       )
                                  ]
                         ),
                        dict(student_module1='neck.fpn_convs1.0.conv',
                             student_module2='neck.fpn_convs2.0.conv',
                             teacher1_module='neck.fpn_convs.0.conv',
                             teacher2_module='neck.fpn_convs.0.conv',
                             output_hook=True,
                             methods=[dict(type='MergeEDLoss',
                                           name='loss_fgd_fpn_0',
                                           student_channels=256,
                                           teacher_channels=256,
                                           temp=temp,
                                           alpha_fgd=alpha_fgd,
                                           beta_fgd=beta_fgd,
                                           gamma_fgd=gamma_fgd,
                                           lambda_fgd=lambda_fgd,
                                           teacher1_center_txt='my_output/collected_center/teacher1_centers.txt',
                                           teacher2_center_txt='my_output/collected_center/teacher2_centers.txt',
                                           roi_extractor=dict(
                                               type='SingleRoIExtractor',
                                               roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                                               out_channels=256,
                                               featmap_strides=[4, 8, 16, 32])
                                           )
                                      ]
                             )

    ]
    )

student_cfg = 'configs/distillers/merge_kd/new_config/faster_rcnn_twins_fpn_r50_1x_coco.py'
teacher1_cfg = 'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
teacher2_cfg = 'configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
dataset_type = 'CocoDataset'
data_root = 'demo/'
runner = dict(type='EpochBasedRunner', max_epochs=16)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='demo/CocoFormatData/annotations/instances_train2017.json',
        img_prefix='demo/CocoFormatData/train2017/',
        ),
    val=dict(
        type='CocoDataset',
        ann_file='demo/CocoFormatData/annotations/instances_val2017.json',
        img_prefix='demo/CocoFormatData/val2017/',
        ),
    test=dict(
        type='CocoDataset',
        ann_file='demo/CocoFormatData/annotations/instances_val2017.json',
        img_prefix='demo/CocoFormatData/val2017/',
        ))
