# Show Attend and Tell，在Show and Tell基础上引入注意力机制

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import pickle
from utils.beamsearch import beam_search
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Resnet152_Encoder(nn.Module):

    def __init__(self, config):
        super(Resnet152_Encoder, self).__init__()
        self.config = config
        if config.mode == 'train':
            resnet152 = models.resnet152()
            pre_train = torch.load('/home/data_ti4_c/chengkz/data/resnet152-b121ed2d.pth')
            resnet152.load_state_dict(pre_train)
        else:
            resnet152 = models.resnet152(pretrained=False)
        self.cnn = nn.Sequential(*list(resnet152.children())[:-2])  # 输出维度[b,2048,7,7]
        self.avgpool = nn.AvgPool2d(7)  # 平均池化，用于生成global image feature
        for p in self.cnn.parameters():
            p.requires_grad = False     # 训练过程不微调CNN
        # self.fine_tune()

    def fine_tune(self):
        for c in list(self.cnn.children())[7:]:
            for p in c.parameters():
                p.requires_grad = True

    def forward(self, image):
        features = self.cnn(image)
        batch_size = features.shape[0]
        global_features = self.avgpool(features).view(batch_size, -1)  # 全局特征[b,2048]
        spatial_features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, 2048)   # 空间特征[b,49,2048]
        return global_features, spatial_features  # 最终生成的深度为2048的全局特征和49*2048的空间特征

# Attention：注意力机制
# 在生成的每个时刻，根据当前模型的状态决定关注图像中的哪个区域（也就是把图像features map按照怎样的加权输入）
class Spatial_Attention(nn.Module):

    def __init__(self, config):
        super(Spatial_Attention, self).__init__()
        self.image_dim = config.image_dim
        self.hidden_dim = config.hidden_dim
        self.att_dim = config.att_dim
        self.visual2att = nn.Linear(self.image_dim, self.att_dim)
        self.hidden2att = nn.Linear(self.hidden_dim, self.att_dim)
        self.att2weight = nn.Linear(self.att_dim, 1)

    def forward(self, spatial_features, decoder_out):
        batch_size = spatial_features.shape[0]
        num_pixels = spatial_features.shape[1]
        visual_att = self.visual2att(spatial_features)
        hidden_att = self.hidden2att(decoder_out)
        hidden_att = hidden_att.unsqueeze(1).expand(batch_size, num_pixels, self.att_dim)

        att_alpha = self.att2weight(torch.tanh(visual_att+hidden_att)).squeeze(2)
        att_weights = F.softmax(att_alpha, dim=1)

        spatial_feature_weighted = (spatial_features * att_weights.unsqueeze(2)).sum(dim=1)
        return spatial_feature_weighted, att_weights


class LSTM_Decoder_Attention(nn.Module):

    def __init__(self, config):
        super(LSTM_Decoder_Attention, self).__init__()
        self.config = config
        self.vocab = pickle.load(open(self.config.vocab, 'rb'))
        self.vocab_size = self.vocab.get_size()
        self.image_dim = config.image_dim
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstmcell = nn.LSTMCell(self.embed_dim, self.hidden_dim)

        self.image2hidden = nn.Linear(self.image_dim, self.hidden_dim)  # 以便图像特征和hidden相加
        self.image2embed = nn.Linear(self.image_dim, self.embed_dim)  # 以便图像特征和embedding相加
        self.spatial_attention = Spatial_Attention(config)

        self.init_h = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))
        self.init_c = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))

        self.fc = weight_norm(nn.Linear(self.hidden_dim, self.vocab_size))
        self.dropout = nn.Dropout(0.5)

        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def image_init_state(self, global_features):
        h = self.init_h(global_features)
        c = self.init_c(global_features)
        return h, c

    def forward(self, global_features, spatial_features, cap, cap_len):
        batch_size = global_features.size(0)
        h, c = self.image_init_state(global_features)
        embeddings = self.embed(cap)
        logit = torch.zeros(batch_size, max(cap_len), self.vocab_size).to(device)

        for t in range(max(cap_len)):
            spatial_features_weighted, att_weights = self.spatial_attention(spatial_features, h)
            visual_hidden = self.image2hidden(spatial_features_weighted)
            visual_embedding = self.image2embed(spatial_features_weighted)
            h, c = self.lstmcell(embeddings[:, t, :]+visual_embedding, (h, c))
            pred = self.fc(self.dropout(h+visual_hidden))
            logit[:, t, :] = pred

        return logit

    def decode_step(self, input_ids, context):
        # batch beamsearch，需要解码器提供单步解码的结果
        # 为适配transformer based模型的情况，input_ids是从生成过程中生成的所有单词
        # context是解码所需要的状态，transformer based模型是编码器的输出，LSTM based模型则包含上一时刻的隐藏层状态
        # input_ids: (batch_size*num_beams, cur_len) context: [h, c], h: (batch_size*num_beams, context_dim)
        # 输出当前时刻的fc层输出结果，和隐藏层状态
        # context是一个list，最后一个位置记录(h, c)，前面记录一些其他所需信息
        spatial_features = context[0]
        hidden = context[-1]
        spatial_features_weighted, att_weights = self.spatial_attention(spatial_features, context[-1][0])
        visual_hidden = self.image2hidden(spatial_features_weighted)
        visual_embedding = self.image2embed(spatial_features_weighted)
        embedding = self.embed(input_ids[:, -1])+visual_embedding
        h, c = self.lstmcell(embedding, hidden)
        pred = self.fc(h+visual_hidden).unsqueeze(1)
        return pred, (h, c)


class SAT(nn.Module):

    def __init__(self, config):
        super(SAT, self).__init__()
        self.config = config
        self.resnet_encoder = Resnet152_Encoder(self.config)
        self.lstm_decoder_att = LSTM_Decoder_Attention(self.config)

    def forward(self, image_feature, cap, cap_len):
        image = image_feature['image']
        global_features, spatial_features = self.resnet_encoder(image)
        logit = self.lstm_decoder_att(global_features, spatial_features, cap, cap_len)

        return logit

    def fine_tune(self):
        self.resnet_encoder.fine_tune()

    def generate_caption_batchbs(self, image_feature):
        image = image_feature['image']
        batch_size = image.shape[0]
        global_features, spatial_features = self.resnet_encoder(image)
        h, c = self.lstm_decoder_att.image_init_state(global_features)
        h = h.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        c = c.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        num_pixels = spatial_features.shape[1]
        spatial_features = spatial_features.repeat(1, self.config.beam_num, 1).view(batch_size*self.config.beam_num, num_pixels, self.config.image_dim)
        captions = beam_search('LSTM', [spatial_features, (h, c)], self.lstm_decoder_att, batch_size, self.config.fixed_len, self.config.beam_num,
                               self.lstm_decoder_att.vocab_size, self.config.length_penalty)
        return captions
