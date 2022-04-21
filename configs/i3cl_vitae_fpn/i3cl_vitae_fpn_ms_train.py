_base_ = [
    './i3cl_vitae_fpn_model.py',
    './coco_instance.py',
    './schedule.py',
    './default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
        NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'], 
        embed_dims=[64, 64, 128, 256], 
        token_dims=[64, 128, 256, 512], 
        downsample_ratios=[4, 2, 2, 2],
        NC_depth=[2, 2, 8, 2], 
        NC_heads=[1, 2, 4, 8], 
        RC_heads=[1, 1, 2, 4], 
        mlp_ratio=4., 
        NC_group=[1, 32, 64, 128], 
        RC_group=[1, 16, 32, 64],
        use_checkpoint=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        window_size=7,
        pretrained_checkpoint='pretrained_model/ViTAE/model_best.pth.tar'
    ),
    neck=dict(in_channels=[64, 128, 256, 512]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(policy='step', warmup='linear', warmup_iters=1000, warmup_ratio=0.001, step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=12)
find_unused_parameters=True

# resume_from = '...'
# load_from = '...'
