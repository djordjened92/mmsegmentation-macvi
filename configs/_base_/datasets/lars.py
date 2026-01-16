# dataset settings
dataset_type = 'LaRSDataset'
data_root = '/home/djordje/Documents/Projects/MaCVi2026/data/images/'
ignore_idx = 255

# Normalization: Already matches ImageNet protocol
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Target Resolution: 768 (W) x 384 (H)
# For RandomCrop/Pad, we use (H, W)
crop_size = (384, 768) 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Resize to the target scale while allowing for augmentation scaling
    dict(type='Resize', img_scale=(768, 384), ratio_range=(0.5, 2.0)),
    
    ### Augmentations ###
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),

    # Color Jitter (PhotoMetricDistortion is the standard MMSeg equivalent)
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),

    # Affine Transform & Rotation (Combined in one op)
    dict(type='RandomRotate', prob=0.5, degree=15, pad_val=0, seg_pad_val=255),

    # # Custom Unsharp Masking (See code below to register this)
    # dict(type='UnsharpMasking', amount=1.0, threshold=0, prob=0.2),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='CenterPad', size=crop_size, pad_val=0, seg_pad_val=ignore_idx),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 384), # Fixed size for ONNX/SNPE compatibility
        flip=False,
        transforms=[
            # keep_ratio=True maintains aspect ratio as per protocol
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='CenterPad', size=crop_size, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16, # TakuNet is light, you can likely increase this
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root + 'train',
        split='image_list.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root + 'val',
        split='image_list.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root + 'test',
        split='image_list.txt',
        pipeline=test_pipeline))