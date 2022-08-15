log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='SDMGR',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='SDMGRHead', visual_dim=16, num_chars=92, num_classes=4),
    visual_modality=False,
    train_cfg=None,
    test_cfg=None,
    class_list=None,
    openset=True)
optimizer = dict(type='Adam', weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1,
    step=[40, 50])
total_epochs = 60
train_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_texts'))
]
test_pipeline = [
    dict(type='LoadAnnotations'),
    dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
    dict(type='KIEFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'relations', 'texts', 'gt_bboxes'],
        meta_keys=('filename', 'ori_filename', 'ori_texts', 'ori_bboxes',
                   'img_norm_cfg', 'ori_filename', 'img_shape'))
]
dataset_type = 'OpensetKIEDataset'
data_root = 'tests/dataset'
loader = dict(
    type='AnnFileLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))
train = dict(
    type='OpensetKIEDataset',
    ann_file='tests/dataset/train/train.txt',
    pipeline=[
        dict(type='LoadAnnotations'),
        dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
        dict(type='KIEFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'],
            meta_keys=('filename', 'ori_filename', 'ori_texts'))
    ],
    img_prefix='tests/dataset/train/',
    link_type='one-to-many',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='tests/dataset/dict.txt',
    test_mode=False)
test = dict(
    type='OpensetKIEDataset',
    ann_file='tests/dataset/test/test.txt',
    pipeline=[
        dict(type='LoadAnnotations'),
        dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
        dict(type='KIEFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'relations', 'texts', 'gt_bboxes'],
            meta_keys=('filename', 'ori_filename', 'ori_texts', 'ori_bboxes',
                       'img_norm_cfg', 'ori_filename', 'img_shape'))
    ],
    img_prefix='tests/dataset/test/',
    link_type='one-to-many',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    dict_file='tests/dataset/dict.txt',
    test_mode=True)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='OpensetKIEDataset',
        ann_file='tests/dataset/train/train.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes', 'gt_labels'],
                meta_keys=('filename', 'ori_filename', 'ori_texts'))
        ],
        img_prefix='tests/dataset/train/',
        link_type='one-to-many',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file='tests/dataset/dict.txt',
        test_mode=False),
    val=dict(
        type='OpensetKIEDataset',
        ann_file='tests/dataset/test/test.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes'],
                meta_keys=('filename', 'ori_filename', 'ori_texts',
                           'ori_bboxes', 'img_norm_cfg', 'ori_filename',
                           'img_shape'))
        ],
        img_prefix='tests/dataset/test/',
        link_type='one-to-many',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file='tests/dataset/dict.txt',
        test_mode=True),
    test=dict(
        type='OpensetKIEDataset',
        ann_file='tests/dataset/test/test.txt',
        pipeline=[
            dict(type='LoadAnnotations'),
            dict(type='ResizeNoImg', img_scale=(1024, 512), keep_ratio=True),
            dict(type='KIEFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'relations', 'texts', 'gt_bboxes'],
                meta_keys=('filename', 'ori_filename', 'ori_texts',
                           'ori_bboxes', 'img_norm_cfg', 'ori_filename',
                           'img_shape'))
        ],
        img_prefix='tests/dataset/test/',
        link_type='one-to-many',
        loader=dict(
            type='AnnFileLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        dict_file='tests/dataset/dict.txt',
        test_mode=True))
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='openset_f1', metric_options=None)
find_unused_parameters = True
work_dir = './work_dirs/sdmgr_novisual_60e_wildreceipt_openset'
gpu_ids = [0]
