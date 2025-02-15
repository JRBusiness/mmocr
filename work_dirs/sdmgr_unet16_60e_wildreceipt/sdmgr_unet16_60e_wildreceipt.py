img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale = 1024
min_scale = 512
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes'],
        meta_keys=[
            'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
            'ori_texts'
        ])
]
dataset_type = 'KIEDataset'
data_root = 'tests/wildreceipt/'
loader = dict(
    type='AnnFileLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))
train = dict(
    type='KIEDataset',
    ann_file='tests/wildreceipt/train.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='KIEFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
    ],
    img_prefix='tests/wildreceipt/image-files/',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='tests/wildreceipt/dict.txt',
    test_mode=False)
test = dict(
    type='KIEDataset',
    ann_file='tests/wildreceipt/test.txt',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='KIEFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'relations', 'texts', 'gt_bboxes'],
            meta_keys=[
                'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
                'ori_texts'
            ])
    ],
    img_prefix='tests/wildreceipt/image_files/',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='tests/wildreceipt/dict.txt',
    test_mode=True)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='KIEDataset',
        ann_file='tests/wildreceipt/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'])
        ],
        img_prefix='tests/wildreceipt/image_files/',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file='tests/wildreceipt/dict.txt',
        test_mode=False),
    val=dict(
        type='KIEDataset',
        ann_file='tests/wildreceipt/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes'],
                meta_keys=[
                    'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
                    'ori_texts'
                ])
        ],
        img_prefix='tests/wildreceipt/image_files/',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file='tests/wildreceipt/dict.txt',
        test_mode=True),
    test=dict(
        type='KIEDataset',
        ann_file='tests/wildreceipt/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes'],
                meta_keys=[
                    'img_norm_cfg', 'img_shape', 'ori_filename', 'filename',
                    'ori_texts'
                ])
        ],
        img_prefix='tests/wildrecept/image_files/',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file='tests/wildreceipt/dict.txt',
        test_mode=True))
evaluation = dict(
    interval=1,
    metric='macro_f1',
    metric_options=dict(
        macro_f1=dict(
            ignores=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])))
model = dict(
    type='SDMGR',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='SDMGRHead', visual_dim=16, num_chars=92, num_classes=26),
    visual_modality=True,
    train_cfg=None,
    test_cfg=None,
    class_list='tests/wildreceipt/class_list.txt')
optimizer = dict(type='Adam', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1,
    step=[40, 50])
total_epochs = 60
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
work_dir = './work_dirs/sdmgr_unet16_60e_wildreceipt'
gpu_ids = [0]
