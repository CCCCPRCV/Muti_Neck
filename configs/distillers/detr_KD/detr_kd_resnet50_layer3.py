_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_2x.py', '../../_base_/default_runtime.py'
]
#dataset setting


# model settings



find_unused_parameters=True
temp=0.5
alpha_fgd=0.001
beta_fgd=0.0005
gamma_fgd=0.0005
lambda_fgd=0.000005
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
    init_student = True,
    student_pretrain = '/mnt/Data2/cjc/Py_Project/year2022/KD_KnowledgeDistillation/mmdetection_Test_FGD_DETR/my_output/detr_resnet50_layer3_pre.pth',
    distill_cfg = [
                    # dict(student_module = 'backbone.layer4',
                    #      teacher_module = 'backbone.layer4',
                    #      output_hook = True,
                    #      methods=[dict(type='FeatureLoss',
                    #                    name='resnet_feature_loss',
                    #                    student_channels = 2048,
                    #                    teacher_channels = 2048,
                    #                    temp = temp,
                    #                    alpha_fgd=alpha_fgd,
                    #                    beta_fgd=beta_fgd,
                    #                    gamma_fgd=gamma_fgd,
                    #                    lambda_fgd=lambda_fgd,
                    #                    )
                    #             ]
                    #     ),
                    dict(student_module = 'backbone.layer4',
                         teacher_module = 'backbone.layer4',
                         output_hook = True,
                         methods=[dict(type='DecoderFeatureLoss',
                                       name='resnet_feature_loss',
                                       ratio = 1,
                                       )
                                ]
                        ),
                    dict(student_module = 'bbox_head.transformer.encoder.layers.0',
                         teacher_module = 'bbox_head.transformer.encoder.layers.1',
                         output_hook = True,
                         methods=[dict(type='EncoderFeatureLoss',
                                       name='transformer_encoderlayer0_loss',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       ratio = 1,
                                       )
                                ]
                        ),
                    dict(student_module = 'bbox_head.transformer.encoder.layers.1',
                         teacher_module = 'bbox_head.transformer.encoder.layers.3',
                         output_hook = True,
                         methods=[dict(type='EncoderFeatureLoss',
                                       name='transformer_encoderlayer1_loss',
                                       student_channels=256,
                                       teacher_channels=256,
                                       ratio = 1,
                                       )
                                ]
                        ),
                    dict(student_module = 'bbox_head.transformer.encoder.layers.2',
                         teacher_module = 'bbox_head.transformer.encoder.layers.5',
                         output_hook = True,
                         methods=[dict(type='EncoderFeatureLoss',
                                       name='transformer_encoderlayer2_loss',
                                       student_channels=256,
                                       teacher_channels=256,
                                       ratio = 1,
                                       )
                                ]
                        ),
                    # dict(student_module = 'bbox_head.transformer.decoder.layers.0',
                    #      teacher_module = 'bbox_head.transformer.decoder.layers.1',
                    #      output_hook = True,
                    #      methods=[dict(type='DecoderFeatureLoss',
                    #                    name='transformer_decoderlayer0_loss',
                    #                    ratio=0.5
                    #                    )
                    #             ]
                    #     ),
                    # dict(student_module = 'bbox_head.transformer.decoder.layers.1',
                    #      teacher_module = 'bbox_head.transformer.decoder.layers.3',
                    #      output_hook = True,
                    #      methods=[dict(type='DecoderFeatureLoss',
                    #                    name='transformer_decoderlayer1_loss',
                    #                    ratio=0.5
                    #                    )
                    #             ]
                    #     ),
                    # dict(student_module = 'bbox_head.transformer.decoder.layers.2',
                    #      teacher_module = 'bbox_head.transformer.decoder.layers.5',
                    #      output_hook = True,
                    #      methods=[dict(type='DecoderFeatureLoss',
                    #                    name='transformer_decoderlayer2_loss',
                    #                    ratio=0.5
                    #                    )
                    #             ]
                    #     ),

                   ]
    )

student_cfg = 'configs/DETR-Small/detr_resnet50_layer3.py'
teacher_cfg = 'configs/detr/detr_r50_8x2_150e_coco.py'
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=150)
dataset_type = 'CocoDataset'
data_root = '/mnt/Data2/cjc/Py_Project/DataSet/generic_detection/COCO_cat_dog/COCOFormatData/'
data = dict(
    samples_per_gpu=2,
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
