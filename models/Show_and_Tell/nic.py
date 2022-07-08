# Show and Tell，以resnet152作为图像编码器，LSTM作为解码器

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
        return global_features  # 最终生成的深度为2048的全局特征


class LSTM_Decoder(nn.Module):

    def __init__(self, config):
        super(LSTM_Decoder, self).__init__()
        self.config = config
        self.vocab = pickle.load(open(self.config.vocab, 'rb'))
        self.vocab_size = self.vocab.get_size()
        self.image_dim = config.image_dim
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstmcell = nn.LSTMCell(self.embed_dim, self.hidden_dim)

        self.init_h = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))
        self.init_c = weight_norm(nn.Linear(self.image_dim, self.hidden_dim))

        self.fc = weight_norm(nn.Linear(self.hidden_dim, self.vocab_size))
        self.dropout = nn.Dropout(0.5)

        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def image_init_state(self, img_feat):
        h = self.init_h(img_feat)
        c = self.init_c(img_feat)
        return h, c

    def forward(self, img_feat, cap, cap_len):
        batch_size = img_feat.size(0)
        h, c = self.image_init_state(img_feat)
        embeddings = self.embed(cap)
        logit = torch.zeros(batch_size, max(cap_len), self.vocab_size).to(device)

        for t in range(max(cap_len)):
            h, c = self.lstmcell(embeddings[:, t, :], (h, c))
            pred = self.fc(self.dropout(h))
            logit[:, t, :] = pred

        return logit

    def decode_step(self, input_ids, context):
        # batch beamsearch，需要解码器提供单步解码的结果
        # 为适配transformer based模型的情况，input_ids是从生成过程中生成的所有单词
        # context是解码所需要的状态，transformer based模型是编码器的输出，LSTM based模型则是上一时刻的隐藏层状态
        # input_ids: (batch_size*num_beams, cur_len) context: [h, c], h: (batch_size*num_beams, hidden_dim)
        # 输出当前时刻的fc层输出结果，和隐藏层状态
        hidden = context[-1]
        embedding = self.embed(input_ids[:, -1])
        h, c = self.lstmcell(embedding, hidden)
        pred = self.fc(h).unsqueeze(1)
        return pred, (h, c)

    def beam_search(self, img_feat):
        beam_num = self.config.beam_num

        sample = []
        sample_score = []
        live_k = 1
        dead_k = 0

        hyp_samples = [[1]] * live_k
        hyp_scores = torch.zeros(1).to(device)

        h, c = self.image_init_state(img_feat)

        hyp_status = [0]
        hyp_status[0] = (h, c)

        for i in range(self.config.fixed_len+1):
            sen_size = len(hyp_samples)

            embedding = [hyp_samples[j][-1] for j in range(sen_size)]
            embedding = torch.Tensor(embedding).long().to(device)
            embedding = self.embed(embedding)

            h_batch = torch.cat([hyp_status[j][0] for j in range(sen_size)], dim=0)
            c_batch = torch.cat([hyp_status[j][1] for j in range(sen_size)], dim=0)

            h, c = self.lstmcell(embedding, (h_batch, c_batch))
            pred = self.fc(h)
            probs = F.softmax(pred, 1)

            can_score = hyp_scores.expand([self.vocab_size, sen_size]).permute(1, 0)
            can_score = can_score-torch.log(probs)
            can_score_flat = can_score.flatten()
            word_ranks = can_score_flat.argsort()[:(beam_num-dead_k)]
            status_indices = torch.floor_divide(word_ranks, self.vocab_size)
            word_indices = word_ranks % self.vocab_size
            after_score = can_score_flat[word_ranks]

            new_hyp_samples = []
            new_hyp_scores = []
            new_hyp_status = []
            live_k = 0
            for idx, [si, wi] in enumerate(zip(status_indices, word_indices)):
                if int(wi) == 2:
                    sample.append(hyp_samples[si]+[int(wi)])
                    sample_score.append((after_score[idx]))
                    dead_k += 1
                else:
                    live_k += 1
                    new_hyp_samples.append(hyp_samples[si]+[int(wi)])
                    new_hyp_scores.append(after_score[idx])
                    new_hyp_status.append((h[si].unsqueeze(0), c[si].unsqueeze(0)))

            hyp_samples = new_hyp_samples
            hyp_scores = torch.Tensor(new_hyp_scores).to(device)
            hyp_status = new_hyp_status

            if live_k < 1:
                break

        for i in range(len(hyp_samples)):
            sample.append(hyp_samples[i])
            sample_score.append(hyp_scores[i])

        alpha = self.config.beam_alpha
        for j in range(len(sample_score)):
            sample_score[j] = sample_score[j]/(pow((5+len(sample[j])), alpha)/pow(5+1, alpha))

        # 将sample中的beam_num个句子按照得分从高到低（log从低到高）排序
        rank = [item[0] for item in sorted(enumerate(sample_score), key=lambda x:x[1])]
        sample_final = [sample[item] for item in rank]

        min_id = sample_score.index(min(sample_score))

        return [sample[min_id]], sample_final


class NIC(nn.Module):

    def __init__(self, config):
        super(NIC, self).__init__()
        self.config = config
        self.resnet_encoder = Resnet152_Encoder(self.config)
        self.lstm_decoder = LSTM_Decoder(self.config)

    def forward(self, image_feature, cap, cap_len):
        image = image_feature['image']
        img_feat = self.resnet_encoder(image)
        logit = self.lstm_decoder(img_feat, cap, cap_len)

        return logit

    def fine_tune(self):
        self.resnet_encoder.fine_tune()

    def generate_caption(self, image_feature):
        image = image_feature['image']
        img_feat = self.resnet_encoder(image)
        caption, captions = self.lstm_decoder.beam_search(img_feat)

        return caption

    def generate_caption_batchbs(self, image):
        batch_size = image.shape[0]
        img_feat = self.resnet_encoder(image)
        h, c = self.lstm_decoder.image_init_state(img_feat)
        h = h.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        c = c.repeat(1, self.config.beam_num).view(batch_size*self.config.beam_num, -1)
        captions = beam_search('LSTM', [(h, c)], self.lstm_decoder, batch_size, self.config.fixed_len, self.config.beam_num,
                               self.lstm_decoder.vocab_size, self.config.length_penalty)
        return captions


