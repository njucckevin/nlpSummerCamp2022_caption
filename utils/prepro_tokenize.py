# 将AI_Challenge数据集中给出的描述句子按照jieba分词tokenize
# 并构建用于训练、验证和测试的json格式数据集，以便dataloader直接处理

import os
import json
from tqdm import tqdm
import jieba

def tokenize(sent):
    return list(jieba.cut(sent.strip().replace(u'。', '').replace(' ', ''), cut_all=False))

data_all = {'train': '../data/caption_train_annotations_20170902.json',
            'val': '../data/caption_validation_annotations_20170910.json',
            'testa': '../data/caption_test_a_annotations_20180103.json',
            'testb': '../data/caption_test_b_annotations_20180103.json'}

print("Processing train...")
data_raw_train = json.load(open(data_all['train'], 'r'))
images_dir_train = '/home/nlper_data/chengkz/AI_Challenge/train_images'
data_train = []
for i, item in tqdm(enumerate(data_raw_train)):
    filename = os.path.join(images_dir_train, item['image_id'])
    image_id = item['image_id'][:-4]
    for sentence in item['caption']:
        sent_tokens = tokenize(sentence)
        item_train = {'split': 'train', 'image_id': image_id, 'filename': filename, 'caption': sent_tokens}
        data_train.append(item_train)
print("Num train: "+str(len(data_train)))

print("Processing val...")
data_raw_val = json.load(open(data_all['val'], 'r'))
images_dir_val = '/home/nlper_data/chengkz/AI_Challenge/val_images'
data_val = []
for i, item in tqdm(enumerate(data_raw_val)):
    filename = os.path.join(images_dir_val, item['image_id'])
    image_id = item['image_id'][:-4]
    captions = [tokenize(sentence) for sentence in item['caption']]
    if len(captions) != 5:
        print('with less than 5 captions')
    item_val = {'split': 'val', 'image_id': image_id, 'filename': filename, 'caption': captions}
    data_val.append(item_val)
print("Num val: "+str(len(data_val)))

print("Processing test...")
id_to_imageid = {}
for item in json.load(open(data_all['testa'], 'r'))['images']:
    id_to_imageid[item['id']] = item['file_name']
data_raw_testa = json.load(open(data_all['testa'], 'r'))['annotations']
images_dir_testa = '/home/nlper_data/chengkz/AI_Challenge/testa_images'
data_testa = []
for i in tqdm(range(30000)):
    filename = os.path.join(images_dir_testa, id_to_imageid[data_raw_testa[i*5]['image_id']]+'.jpg')
    image_id = id_to_imageid[data_raw_testa[i*5]['image_id']]
    captions = []
    for item in data_raw_testa[i*5: i*5+5]:
        if item['image_id'] != data_raw_testa[i*5]['image_id']:
            print('5 captions not belong to the same image')
        captions.append(tokenize(item['caption']))
    item_testa = {'split': 'testa', 'image_id': image_id, 'filename': filename, 'caption': captions}
    data_testa.append(item_testa)
print("Num testa: "+str(len(data_testa)))

id_to_imageid = {}
for item in json.load(open(data_all['testb'], 'r'))['images']:
    id_to_imageid[item['id']] = item['file_name']
data_raw_testb = json.load(open(data_all['testb'], 'r'))['annotations']
images_dir_testb = '/home/nlper_data/chengkz/AI_Challenge/testb_images'
data_testb = []
for i in tqdm(range(30000)):
    filename = os.path.join(images_dir_testb, id_to_imageid[data_raw_testb[i*5]['image_id']]+'.jpg')
    image_id = id_to_imageid[data_raw_testb[i*5]['image_id']]
    captions = []
    for item in data_raw_testb[i*5: i*5+5]:
        if item['image_id'] != data_raw_testb[i*5]['image_id']:
            print('5 captions not belong to the same image')
        captions.append(tokenize(item['caption']))
    item_testb = {'split': 'testb', 'image_id': image_id, 'filename': filename, 'caption': captions}
    data_testb.append(item_testb)
print("Num testb: "+str(len(data_testb)))

json.dump(data_train, open('../data/train.json', 'w'), ensure_ascii=False)
json.dump(data_val, open('../data/val.json', 'w'), ensure_ascii=False)
json.dump(data_testa, open('../data/testa.json', 'w'), ensure_ascii=False)
json.dump(data_testb, open('../data/testb.json', 'w'), ensure_ascii=False)


