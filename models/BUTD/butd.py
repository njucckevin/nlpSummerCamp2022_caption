# Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering，使用faster-rcnn提取的目标检测特征作为视觉输入，并设计了包含Top-Down Attention LSTM和Language LSTM解码器
# 此后，以目标检测特征作为视觉输入被许多跨模态任务广泛采用

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import pickle
from utils.beamsearch import beam_search
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BUTD_Attend(nn.Module):
    """
    Attention Network in TopDownDecoder
    """
    def __init__(self, fea_dim, hid_dim, att_dim, dropout=0.5):
        super(BUTD_Attend, self).__init__()

        self.fea_dim = fea_dim
        self.hid_dim = hid_dim
        self.att_dim = att_dim

        # three layers for calculate weight alpha
        self.fea_att = weight_norm(nn.Linear(self.fea_dim, self.att_dim))
        self.hid_att = weight_norm(nn.Linear(self.hid_dim, self.att_dim))
        self.att_weight = weight_norm(nn.Linear(self.att_dim, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, fea_maps, Attention_lstm_hidden):
        """
        :param fea_maps: feature of image (batch_size, 36, fea_dim)
        :param Attention_lstm_hidden: h of Attention LSTM (batch_size, hid_dim)
        :return: weighted sum of the fea_maps (batch_size, fea_dim), weights
        """
        att1 = self.fea_att(fea_maps)  # (batch_size, 36, att_dim)
        att2 = self.hid_att(Attention_lstm_hidden)  # (batch_size, att_dim)
        att = self.att_weight(self.dropout(self.relu(att1+att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        weight = F.softmax(att, 1)  # (batch_size, 36)
        weighted_fea = (fea_maps * weight.unsqueeze(2)).sum(dim=1)  # (batch_size, fea_dim)

        return weighted_fea, weight


class TopDownDecoder(nn.Module):

    def __init__(self, config):
        super(TopDownDecoder, self).__init__()
        self.config = config
        self.vocab = pickle.load(open(self.config.vocab, 'rb'))
        self.vocab_size = self.vocab.get_size()
        self.image_dim = config.image_dim
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim
        self.att_dim = config.att_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.Topdown_Attention_lstmcell = nn.LSTMCell(self.hidden_dim+self.image_dim+self.embed_dim, self.hidden_dim, bias=True)
        self.Attend = BUTD_Attend(self.image_dim, self.hidden_dim, self.att_dim)
        self.Language_lstmcell = nn.LSTMCell(self.image_dim+self.hidden_dim, self.hidden_dim, bias=True)

        self.fc = weight_norm(nn.Linear(self.hidden_dim, self.vocab_size))
        self.dropout = nn.Dropout(0.5)

        self.init_h_A = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))
        self.init_c_A = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))
        self.init_h_L = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))
        self.init_c_L = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))
        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def fea_init_state(self, fea_vec):
        '''use image features(means) to initiate state of two LSTM'''
        h_A = self.init_h_A(fea_vec)
        c_A = self.init_c_A(fea_vec)
        h_L = self.init_h_L(fea_vec)
        c_L = self.init_c_L(fea_vec)
        return h_A, c_A, h_L, c_L

    def forward(self, fea_vec, fea_maps, cap, cap_len):
        batch_size = fea_maps.size(0)
        h_A, c_A, h_L, c_L = self.fea_init_state(fea_vec)   # (batch_size, hid_dim)
        embeddings = self.embed(cap)
        logit = torch.zeros(batch_size, max(cap_len), self.vocab_size).to(device)

        for t in range(max(cap_len)):
            input_attention_lstm = torch.cat([h_L, fea_vec, embeddings[:, t, :]], dim=1)
            h_A, c_A = self.Topdown_Attention_lstmcell(input_attention_lstm, (h_A, c_A))
            weighted_fea, weight = self.Attend(fea_maps, h_A)
            input_language_lstm = torch.cat([weighted_fea, h_A], dim=1)
            h_L, c_L = self.Language_lstmcell(input_language_lstm, (h_L, c_L))
            pred = self.fc(self.dropout(h_L))
            logit[:, t, :] = pred

        return logit

    def decode_step(self, input_ids, context):
        fea_maps = context[0]
        fea_vec = fea_maps.mean(dim=1)
        h_A, c_A, h_L, c_L = context[-1]
        embedding = self.embed(input_ids[:, -1])

        input_attention_lstm = torch.cat([h_L, fea_vec, embedding], dim=1)
        h_A, c_A = self.Topdown_Attention_lstmcell(input_attention_lstm, (h_A, c_A))
        weighted_fea, weight = self.Attend(fea_maps, h_A)
        input_language_lstm = torch.cat([weighted_fea, h_A], dim=1)
        h_L, c_L = self.Language_lstmcell(input_language_lstm, (h_L, c_L))
        pred = self.fc(self.dropout(h_L)).unsqueeze(1)
        return pred, (h_A, c_A, h_L, c_L)


class BUTD(nn.Module):

    def __init__(self, config):
        super(BUTD, self).__init__()
        self.config = config
        self.butd_decoder = TopDownDecoder(self.config)

    def forward(self, image_feature, cap, cap_len):
        feature_vec = image_feature['feature_vec']
        feature_map = image_feature['feature_map']
        logit = self.butd_decoder(feature_vec, feature_map, cap, cap_len)

        return logit

    def generate_caption_batchbs(self, image_feature):
        fea_vec = image_feature['feature_vec']
        fea_maps = image_feature['feature_map']
        batch_size = fea_vec.shape[0]
        h_A, c_A, h_L, c_L = self.butd_decoder.fea_init_state(fea_vec)
        h_A = h_A.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        c_A = c_A.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        h_L = h_L.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        c_L = c_L.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)

        num_pixels = fea_maps.shape[1]
        fea_maps = fea_maps.repeat(1, self.config.beam_num, 1).view(batch_size*self.config.beam_num, num_pixels, self.config.image_dim)
        captions = beam_search('LSTM', [fea_maps, (h_A, c_A, h_L, c_L)], self.butd_decoder, batch_size, self.config.fixed_len, self.config.beam_num,
                               self.butd_decoder.vocab_size, self.config.length_penalty)
        return captions
