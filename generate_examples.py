# 为指定文件夹内的图片生成caption，并保存在excel中
# 给出模型名称、model参数和生成结果存储位置
# python generate_examples.py --model NIC --mode test --pretrain_model ./checkpoints/AdaAtt/model_90000.pt --samples_out /examples/examples_out.xlsx

from tqdm import tqdm
import xlsxwriter
import os

import torch
import numpy as np
import pickle

from config import config
from utils.import_models import construct_model

from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = construct_model(config).to(device)
model.load_state_dict(torch.load(config.pretrain_model, map_location=device))
model.eval()

vocab = pickle.load(open(config.vocab, 'rb'))

img_list = [file for file in os.listdir(config.samples_dir) if os.path.splitext(file)[1] == '.jpg']
feat_list = [file for file in os.listdir(config.samples_dir) if os.path.splitext(file)[1] == '.npz']
print("Num of images: "+str(len(img_list)))

book = xlsxwriter.Workbook(config.samples_out)
sheet = book.add_worksheet("samples")
sheet.write("A1", "Image")
sheet.write("B1", "Caption")
sheet.set_column("A:A", 20)
sheet.set_column("B:B", 60)

transform = transforms.Compose([
                 transforms.Resize([224, 224], InterpolationMode.LANCZOS),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))])

for i, image in tqdm(enumerate(img_list)):
    image_path = os.path.join(config.samples_dir, image)
    if config.model in ['NIC', 'SAT', 'AdaAtt']:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        image_feature = {'image': image}
    elif config.model == 'BUTD':
        feature_map = torch.Tensor(np.load(os.path.join(config.samples_dir, image+'.npz'))['feat'])
        feature_vec = feature_map.mean(dim=0)
        if feature_map.shape[0] <= 36:
            pad = torch.zeros(36 - feature_map.shape[0], 2048)
            feature_map = torch.cat([feature_map, pad], dim=0)
        else:
            feature_map = feature_map[:36, :]
        image_feature = {'feature_vec': feature_vec.unsqueeze(0), 'feature_map': feature_map.unsqueeze(0)}
        image_feature = {k: v.to(device) for k, v in image_feature.items()}
    cap_id = model.generate_caption_batchbs(image_feature)[0]
    caption = vocab.idList_to_sent(cap_id).replace(' ', '')
    sheet.set_row(i + 1, 120)
    sheet.insert_image("A{}".format(i+2), image_path, {"x_scale": 0.1, "y_scale": 0.1})
    sheet.write("B{}".format(i+2), caption)

book.close()


