# 用于训练的dataloader

import torch
import numpy as np
import json
import os
import pickle
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IC_data(Dataset):

    def __init__(self, config, dir, mode):
        super(IC_data, self).__init__()
        self.config = config
        self.data = json.load(open(dir, 'r'))
        self.img_dir = os.path.join(config.image_dir, mode+'_images')
        self.transform = transforms.Compose([
                 transforms.Resize([224, 224], InterpolationMode.LANCZOS),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))])
        self.vocab = pickle.load(open(self.config.vocab, 'rb'))
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == 'train':
            cap_token = self.data[item]['caption']
            cap_id, cap_len = self.vocab.tokenList_to_idList(cap_token, self.config.fixed_len)
            image_path = os.path.join(self.img_dir, self.data[item]['image_id']+'.jpg')
            image = self.transform(Image.open(image_path).convert('RGB'))
            return image, torch.Tensor(cap_id).long(), cap_len
        else:
            image_path = os.path.join(self.img_dir, self.data[item]['image_id']+'.jpg')
            image_id = self.data[item]['image_id']
            image = self.transform(Image.open(image_path).convert('RGB'))
            return image_id, image

    def collate_fn_train(self, batch_data):
        image, cap_id, cap_len = zip(*batch_data)
        image = torch.stack(image, dim=0)
        image_feature = {'image': image}
        cap_id = torch.stack(cap_id, dim=0)
        cap_len = torch.Tensor(cap_len).int()
        return image_feature, cap_id, cap_len

    def collate_fn_eval(self, batch_data):
        image_id, image = zip(*batch_data)
        image = torch.stack(image, dim=0)
        image_feature = {'image': image}
        return image_id, image_feature

    def __len__(self):
        return len(self.data)


class IC_data_feature(Dataset):

    def __init__(self, config, dir, mode):
        super(IC_data_feature, self).__init__()
        self.config = config
        self.data = json.load(open(dir, 'r'))
        fasterrcnn_feature_dir = mode+'_fasterrcnn' if (mode == 'testa' or mode == 'testb') else 'trainval_fasterrcnn'
        self.img_feature_dir = os.path.join(config.image_dir, fasterrcnn_feature_dir)
        self.vocab = pickle.load(open(self.config.vocab, 'rb'))
        self.mode = mode

    def __getitem__(self, item):
        if self.mode == 'train':
            cap_token = self.data[item]['caption']
            cap_id, cap_len = self.vocab.tokenList_to_idList(cap_token, self.config.fixed_len)
            feature_map = torch.Tensor(np.load(os.path.join(self.img_feature_dir, self.data[item]['image_id']+'.jpg.npz'))['feat'])
            feature_vec = feature_map.mean(dim=0)
            if feature_map.shape[0] <= 36:
                pad = torch.zeros(36-feature_map.shape[0], 2048)
                feature_map = torch.cat([feature_map, pad], dim=0)
            else:
                feature_map = feature_map[:36, :]
            return feature_vec, feature_map, torch.Tensor(cap_id).long(), cap_len
        else:
            feature_map = torch.Tensor(np.load(os.path.join(self.img_feature_dir, self.data[item]['image_id']+'.jpg.npz'))['feat'])
            feature_vec = feature_map.mean(dim=0)
            if feature_map.shape[0] <= 36:
                pad = torch.zeros(36-feature_map.shape[0], 2048)
                feature_map = torch.cat([feature_map, pad], dim=0)
            else:
                feature_map = feature_map[:36, :]
            image_id = self.data[item]['image_id']
            return image_id, feature_vec, feature_map

    def collate_fn_train(self, batch_data):
        feature_vec, feature_map, cap_id, cap_len = zip(*batch_data)
        feature_vec = torch.stack(feature_vec, dim=0)
        feature_map = torch.stack(feature_map, dim=0)
        image_feature = {'feature_vec': feature_vec, 'feature_map': feature_map}
        cap_id = torch.stack(cap_id, dim=0)
        cap_len = torch.Tensor(cap_len).int()
        return image_feature, cap_id, cap_len

    def collate_fn_eval(self, batch_data):
        image_id, feature_vec, feature_map = zip(*batch_data)
        feature_vec = torch.stack(feature_vec, dim=0)
        feature_map = torch.stack(feature_map, dim=0)
        image_feature = {'feature_vec': feature_vec, 'feature_map': feature_map}
        return image_id, image_feature

    def __len__(self):
        return len(self.data)


def data_load(config, dir, mode):
    if config.image == 'origin':
        dataset = IC_data(config, dir, mode)
    elif config.image == 'fasterrcnn':
        dataset = IC_data_feature(config, dir, mode)
    else:
        dataset = None
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             shuffle=True if mode == 'train' else False,
                             collate_fn=dataset.collate_fn_train if mode == 'train' else dataset.collate_fn_eval,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             )
    return data_loader

def data_load_ddp(config, dir, mode):
    if config.image == 'origin':
        dataset = IC_data(config, dir, mode)
    elif config.image == 'fasterrcnn':
        dataset = IC_data_feature(config, dir, mode)
    else:
        dataset = None
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(dataset=dataset,
                              batch_size=config.batch_size,
                              sampler=data_sampler,
                              collate_fn=dataset.collate_fn_train if mode == 'train' else dataset.collate_fn_eval,
                              num_workers=config.num_workers,
                              )
    return data_loader


