# _base_ = [
#     '../../_base_/datasets/coco_detection.py',
#     '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
# ]
_base_ = [
    '../../../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]
find_unused_parameters=True
temp=0.8
alpha_fgd=0.001
beta_fgd=0.0005
gamma_fgd=0.0005
lambda_fgd=0.000005
teacher1_similarity = 0.5
distiller = dict(
    type='CenterCollerter',
    teacher1_pretrained = 'my_output/my_pth/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth',
    teacher2_pretrained = 'my_output/my_pth/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth',
    roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
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
                    # dict(
                    #      teacher1_module='neck.fpn_convs.3.conv',
                    #      teacher2_module='neck.fpn_convs.3.conv',
                    #      output_hook=True,
                    #      methods=[dict(type='MergeLoss',
                    #                    name='loss_fgd_fpn_3',
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
                    # dict(
                    #      teacher1_module='neck.fpn_convs.2.conv',
                    #      teacher2_module='neck.fpn_convs.2.conv',
                    #      output_hook=True,
                    #      methods=[dict(type='MergeLoss',
                    #                    name='loss_fgd_fpn_2',
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
                    # dict(
                    #      teacher1_module='neck.fpn_convs.1.conv',
                    #      teacher2_module='neck.fpn_convs.1.conv',
                    #      output_hook=True,
                    #      methods=[dict(type='MergeLoss',
                    #                    name='loss_fgd_fpn_1',
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
                        dict(
                             teacher1_module='neck.fpn_convs.2.conv',
                             teacher2_module='neck.fpn_convs.2.conv',
                             output_hook=True,
                             )

    ]
    )

teacher1_cfg = 'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
teacher2_cfg = 'configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py'
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))




dataset_type = 'CocoDataset'
data_root = 'D:\software_data\Py_project\year2023\Datasets/COCOFormatData/'
runner = dict(type='EpochBasedRunner', max_epochs=1)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    train=dict(
        type='CocoDataset',
        ann_file='demo/CocoFormatData/annotations/instances_train2017.json',
        img_prefix='demo/CocoFormatData/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 800)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='demo/CocoFormatData/annotations/instances_val2017.json',
        img_prefix='demo/CocoFormatData/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='demo/CocoFormatData/annotations/instances_val2017.json',
        img_prefix='demo/CocoFormatData/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
