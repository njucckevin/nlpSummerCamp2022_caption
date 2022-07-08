# 测试模型
# 为验证、测试集生成句子并保存为可用pycoco直接计算指标的格式
# 用保存的句子计算指标

import os
import torch
import pickle
import json

from data_load import data_load
from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_captions(config, model, step, mode):
    print("Generating captions...")

    log_path = config.log_dir.format(config.id)
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    gen_pycoco_path = os.path.join(result_dir, mode+'_'+str(step)+'.json')

    vocab = pickle.load(open(config.vocab, 'rb'))
    data_dir = os.path.join(config.data_dir, mode+'.json')

    eval_loader = data_load(config, data_dir, mode)
    model.eval()
    gen_pycoco = {}

    for i, (image_id, image_feature) in tqdm(enumerate(eval_loader)):
        image_feature = {k: v.to(device) for k, v in image_feature.items()}
        batch_size = len(image_id)
        captions = model.generate_caption_batchbs(image_feature)
        for j, cap_id in enumerate(captions):
            caption = vocab.idList_to_sent(cap_id)
            refs = []
            ref = {'image_id': image_id[j], 'id': i * batch_size + j, 'caption': caption}
            refs.append(ref)
            gen_pycoco[i * config.batch_size + j] = refs

    json.dump(gen_pycoco, open(gen_pycoco_path, 'w'), ensure_ascii=False)

    return gen_pycoco_path


def eval_pycoco(config, gen_pycoco_path, mode):
    print("Calculating pycoco...")
    ref_pycoco_path = os.path.join(config.data_dir, mode+'_pycoco.json')
    ref_pycoco = json.load(open(ref_pycoco_path, 'r'))
    gen_pycoco = json.load(open(gen_pycoco_path, 'r'))

    cocoEval = COCOEvalCap('diy', 'diy')
    pycoco_results = cocoEval.evaluate_diy(ref_pycoco, gen_pycoco)

    return pycoco_results

