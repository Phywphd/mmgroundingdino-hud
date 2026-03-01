_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

data_root = '/home/kinova/ssd1/ljx/my_dataset/'
class_name = ('HUD_area', 'aircraft_cursor', 'x_left', 'y_up', 'y_down')
num_classes = len(class_name)
metainfo = dict(
    classes=class_name, 
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
)

model = dict(bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=2,  # batch size per GPU
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = val_evaluator

max_epoch = 30

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=10))
    
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[20, 27],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),  # 微调backbone，使用较小学习率
            'language_model': dict(lr_mult=0.0)  # 冻结语言模型
        }))

# 加载预训练权重
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'  # noqa

# 自动混合精度训练（可选，可以加速训练）
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'backbone': dict(lr_mult=0.1),
#             'language_model': dict(lr_mult=0.0),
#         }))
