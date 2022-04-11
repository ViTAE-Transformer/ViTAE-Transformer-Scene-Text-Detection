# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_ssl=False),  # with_ssl=True for I3CL+SSL training
    dict(type='RandomFlip', flip_ratio=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='AutoAugment',
             policies=[
                 [
                     dict(type='Rotate', prob=0.5, level=10, max_rotate_angle=10)
                 ],
                 [
                     dict(type='ColorTransform', prob=0.5, level=8),
                     dict(type='BrightnessTransform', prob=0.5, level=8),
                     dict(type='ContrastTransform', prob=0.5, level=8),
                 ],
                 [
                     dict(type='ColorTransform', prob=0.5, level=8),
                     dict(type='BrightnessTransform', prob=0.5, level=8),
                     dict(type='ContrastTransform', prob=0.5, level=8),
                     dict(type='Rotate', prob=0.5, level=10, max_rotate_angle=10)
                 ]
             ]),
    dict(type='Resize',
         img_scale=[(480,1333),(512,1333),(544,1333),(576,1333),(608,1333),(640,1333),
                    (704,1333),(732,1333),(768,1333),(800,1600),(832,1600),(864,1600),(896,1600)],
         multiscale_mode='value', keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1950,1185),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# only 'text' class in annotation actually
classes = ('text','0', '1', '2', '3', '4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
           'P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
           'r','s','t','u','v','w','x','y','z')

art=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art/train.json',
        img_prefix=data_root + 'art/train_images/',
        pipeline=train_pipeline)
art_noise=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art_noise/train.json',
        img_prefix=data_root + 'art_noise/train_images/',
        pipeline=train_pipeline)
art_light=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art_light/train.json',
        img_prefix=data_root + 'art_light/train_images/',
        pipeline=train_pipeline)
art_sig=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art_sig/train.json',
        img_prefix=data_root + 'art_sig/train_images/',
        pipeline=train_pipeline)
lsvt_1=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'lsvt/train1.json',
        img_prefix=data_root + 'lsvt/train_images1/',
        pipeline=train_pipeline)
lsvt_2=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'lsvt/train2.json',
        img_prefix=data_root + 'lsvt/train_images2/',
        pipeline=train_pipeline)
ic19mlt_1=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'icdar2019_mlt/train1.json',
        img_prefix=data_root + 'icdar2019_mlt/train_images1/',
        pipeline=train_pipeline)
ic19mlt_2=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'icdar2019_mlt/train2.json',
        img_prefix=data_root + 'icdar2019_mlt/train_images2/',
        pipeline=train_pipeline)

# data for SSL
art_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art/train_lossweight.json',
        img_prefix=data_root + 'art/train_images/',
        pipeline=train_pipeline)
art_noise_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art_noise/train_lossweight.json',
        img_prefix=data_root + 'art_noise/train_images/',
        pipeline=train_pipeline)
art_light_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art_light/train_lossweight.json',
        img_prefix=data_root + 'art_light/train_images/',
        pipeline=train_pipeline)
art_sig_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'art_sig/train_lossweight.json',
        img_prefix=data_root + 'art_sig/train_images/',
        pipeline=train_pipeline)
lsvt_1_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'lsvt/train1_lossweight.json',
        img_prefix=data_root + 'lsvt/train_images1/',
        pipeline=train_pipeline)
lsvt_2_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'lsvt/train2_lossweight.json',
        img_prefix=data_root + 'lsvt/train_images2/',
        pipeline=train_pipeline)
lsvt_test_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'lsvt_test/train_lossweight.json',
        img_prefix=data_root + 'lsvt_test/train_images/',
        pipeline=train_pipeline)
ic19mlt_1_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'icdar2019_mlt/train1_lossweight.json',
        img_prefix=data_root + 'icdar2019_mlt/train_images1/',
        pipeline=train_pipeline)
ic19mlt_2_lossweight=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'icdar2019_mlt/train2_lossweight.json',
        img_prefix=data_root + 'icdar2019_mlt/train_images2/',
        pipeline=train_pipeline)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        classes=classes,
        datasets=[lsvt_1, lsvt_2, ic19mlt_1, ic19mlt_2, art, art_light, art_noise, art_sig],
        pipeline=train_pipeline), 
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'art/train.json',
        img_prefix=data_root + 'art/train_images/',
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])
