import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Cross_Entropy(nn.Module):
    # 序列形式的交叉熵
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, logit, cap, cap_len):
        target = cap[:, 1:]
        cap_len = cap_len - 1

        target = pack_padded_sequence(target, cap_len.cpu(), batch_first=True, enforce_sorted=False)[0]
        logit = pack_padded_sequence(logit, cap_len.cpu(), batch_first=True, enforce_sorted=False)[0]

        # cross_entropy
        loss_ce = self.ce(logit, target)

        return loss_ce



