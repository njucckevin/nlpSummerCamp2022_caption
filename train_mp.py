# 训练
# 自动混合精度

import torch
import random
import numpy as np

import time
import torch.nn as nn

from config import config
from data_load import data_load

from utils.import_models import construct_model
from utils.loss import Cross_Entropy
from utils.log import Log_Writer, train_print
from utils.eval import generate_captions, eval_pycoco

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机种子
seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# log
writer = Log_Writer(config)
global_step = 0
loss_ce_avg = 0

# data_loader
train_loader = data_load(config, config.train, 'train')

# model
model = construct_model(config).to(device)
if config.step != 0:
    log_path = config.log_dir.format(config.id)
    trained_model_path = log_path + '/model/model_' + str(config.step) + '.pt'
    model.load_state_dict(torch.load(trained_model_path))
    global_step = config.step

# optimizer
if config.model in ['NIC', 'SAT', 'AdaAtt']:
    optimizer = torch.optim.Adam([{'params': model.resnet_encoder.parameters(), 'lr': config.lr_enc},
                                  {'params': model.lstm_decoder_att.parameters(), 'lr': config.learning_rate}],
                                  betas=(0.9, 0.98), eps=1e-9)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)

# mixed precision
scaler = torch.cuda.amp.GradScaler()

# loss
loss_fn = Cross_Entropy()

for epoch in range(config.epochs):
    model.train()
    totel_step = len(train_loader)
    epoch_time = time.time()
    step_time = time.time()

    if config.model in ['NIC', 'SAT', 'AdaAtt']:
        if epoch == config.ft_epoch:
            model.fine_tune()

    for step, (image_feature, cap, cap_len) in enumerate(train_loader):

        global_step += 1
        optimizer.zero_grad()

        image_feature = {k: v.to(device) for k, v in image_feature.items()}
        cap = cap.to(device)
        cap_len = cap_len.to(device)

        with torch.cuda.amp.autocast():

            logit = model(image_feature, cap, cap_len)

            loss_ce = loss_fn(logit, cap, cap_len)

        loss_ce_avg += loss_ce.item()

        scaler.scale(loss_ce).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_value_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)

        scaler.update()

        if global_step % config.save_loss_freq == 0:
            writer.write_tensorboard('loss_ce', loss_ce_avg/config.save_loss_freq, global_step)
            loss_ce_avg = 0

        train_print(loss_ce.item(), step, totel_step, epoch, time.time() - step_time, time.time() - epoch_time)
        step_time = time.time()

        if global_step % config.save_model_freq == 0:
            print("Evaluating...")

            # 保存模型
            writer.save_model(model, global_step)

            # validation
            model.eval()
            gen_pycoco_path = generate_captions(config, model, global_step, 'val')
            pycoco_results = eval_pycoco(config, gen_pycoco_path, 'val')
            writer.write_metrics(pycoco_results, global_step)

            model.train()



