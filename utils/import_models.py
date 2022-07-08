# 根据命令行构建模型

from models.Show_and_Tell.nic import NIC
from models.Show_Attend_Tell.sat import SAT
from models.Adaptive_Attention.ada_att import AdaAtt
from models.BUTD.butd import BUTD


def construct_model(config):
    if config.model == 'NIC':
        model = NIC(config)
    elif config.model == 'SAT':
        model = SAT(config)
    elif config.model == 'AdaAtt':
        model = AdaAtt(config)
    elif config.model == 'BUTD':
        model = BUTD(config)
    else:
        print("model "+str(config.model)+" not found")
        return None
    return model