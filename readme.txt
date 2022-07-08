image caption框架

├── config.py	# 各类参数
├── data	# 数据，原始数据为带有annotations的4个json文件
│   ├── caption_test_a_annotations_20180103.json
│   ├── caption_test_b_annotations_20180103.json
│   ├── caption_train_annotations_20170902.json
│   ├── caption_validation_annotations_20170910.json
│   ├── testa.json
│   ├── testa_pycoco.json
│   ├── testb.json
│   ├── testb_pycoco.json
│   ├── train.json
│   ├── val.json
│   ├── val_pycoco.json
│   └── vocab.pkl
├── data_load.py	# dataloader
├── generate_examples.py	# 为测试样例生成caption
├── models	# 模型
│   ├── Adaptive_Attention
│   │   ├── __init__.py
│   │   └── ada_att.py
│   ├── BUTD
│   │   ├── __init__.py
│   │   └── butd.py
│   ├── Show_Attend_Tell
│   │   ├── __init__.py
│   │   └── sat.py
│   └── Show_and_Tell
│       ├── __init__.py
│       └── nic.py
├── test.py	# 测试
├── train.py	# 三种训练模式：普通、自动混合精度（mp）、分布式+自动混合精度（ddp）
├── train_ddp.py
├── train_mp.py
└── utils	# 辅助代码&工具
    ├── beamsearch.py	# beamsearch
    ├── eval.py		# 为val&test集生成结果并计算指标
    ├── import_models.py	# 导入各类模型
    ├── log.py		# 训练日志记录
    ├── loss.py		# 损失函数
    ├── prepro_ref_pycoco.py	# 将reference转换为便于cocoEval工具计算指标的形式
    ├── prepro_tokenize.py	# 整理原始数据集并完成分词
    └── vocab.py	# 生成单词表

step1：prepro_tokenize.py，整理原始数据集并完成分词，保存为train/val/test.json
step2：vocab.py，构建单词表vocab.pkl
step3：prepro_ref_pycoco.py，将reference转换为便于cocoEval工具计算指标的形式train/val/test_pycoco.json
step4：之后即可以开始训练，eg：CUDA_VISIBLE_DEVICES=0 python train.py --mode train --model AdaAtt --id my_model --batch_size 100；训练时要指定的参数参考config.py、train.py和train_ddp.py；image_dir代表图像（特征）保存的路径，lig_dir代表训练日志和结果保存的路径，路径信息可参考服务器代码的路径
step5：训练过程中可通过利用tensorboard查看训练过程并自行决定何时终止
step6：利用test.py得到在测试集上的结果