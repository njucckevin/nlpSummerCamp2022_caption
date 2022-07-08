# 测试，通过test参数指定在testa还是testb上进行测试

import torch
import random
import numpy as np

from config import config

from utils.import_models import construct_model
from utils.eval import generate_captions, eval_pycoco


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机种子
seed = config.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# model
model = construct_model(config).to(device)
log_path = config.log_dir.format(config.id)
trained_model_path = log_path + '/model/model_' + str(config.step) + '.pt'
model.load_state_dict(torch.load(trained_model_path))
model.eval()

gen_pycoco_path = generate_captions(config, model, config.step, config.test)
pycoco_results = eval_pycoco(config, gen_pycoco_path, config.test)
print(pycoco_results)




