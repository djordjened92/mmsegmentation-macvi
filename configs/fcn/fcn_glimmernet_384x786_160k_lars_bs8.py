_base_ = [
    '../_base_/datasets/lars.py', # Reference the LaRS dataset config in the repo
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='GlimmerNet',
        img_width=768,
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type='FCNHead',
        in_channels=240,
        in_index=3,           # Uses the 1/32 feature map
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,        # LaRS typically uses 3 classes: Obstacle, Water, Sky
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=240,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Override batch size
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4
)