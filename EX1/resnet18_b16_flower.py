_base_ = ['../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',    # 数据配置
    '../_base_/default_runtime.py'            # 默认运行设置
          ]
model = dict(
    head = dict(
        num_classes = 5,
        topk = (1, )
    ),
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',  # noqa
            prefix='backbone'
        )
    )
)

dataset_type = 'ImageNet'

data_processor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=5,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='RandomResizedCrop', scale=224),     # 随机放缩裁剪
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   # 随机水平翻转
    dict(type='PackInputs'),         # 准备图像以及标签
]

test_pipeline = [
    dict(type='LoadImageFromFile'),     # 读取图像
    dict(type='ResizeEdge', scale=256, edge='short'),  # 短边对其256进行放缩
    dict(type='CenterCrop', crop_size=224),     # 中心裁剪
    dict(type='PackInputs'),                 # 准备图像以及标签
]

train_dataloader = dict(
    batch_size=32,                     # 每张GPU的 batchsize
    num_workers=2,                     # 每个GPU的线程数
    dataset=dict(                      # 训练数据集
        type=dataset_type,
        data_root='data/flower_data',
        ann_file='train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),   # 默认采样器
    persistent_workers=True,                             # 是否保持进程，可以缩短每个epoch的准备时间
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root='data/flower_data',
        ann_file='val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

# 定义评估⽅法
evaluation = dict(metric_options={'topk': (1, )})

# 优化器
optim_wrapper = dict(
    # 使用 SGD 优化器来优化参数
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 学习率策略
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# 训练的配置, 迭代 100 个epoch，每一个训练 epoch 后都做验证集评估
# 'by_epoch=True' 默认使用 `EpochBaseLoop`,  'by_epoch=False' 默认使用 `IterBaseLoop`
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# 使用默认的验证循环控制器
val_cfg = dict()
# 使用默认的测试循环控制器
test_cfg = dict()

# 通过默认策略自动缩放学习率，此策略适用于总批次大小 256
# 如果你使用不同的总批量大小，比如 512 并启用自动学习率缩放
# 我们将学习率扩大到 2 倍
auto_scale_lr = dict(base_batch_size=256)


runner = dict(type='EpochBasedRunner', max_epochs=2)
