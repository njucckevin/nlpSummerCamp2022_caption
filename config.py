import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--id', type=str, default='test')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--image', type=str, default='origin')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--test', type=str, default='testa')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ft_epoch', type=int, default=3)
parser.add_argument('--step', type=int, default=0)

parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--nproc_per_node', type=int, default=-1)

parser.add_argument('--image_dir', default='/home/chengkz/AI_Challenge')
parser.add_argument('--data_dir', default='./data')
parser.add_argument('--vocab', default='./data/vocab.pkl')
parser.add_argument('--train', default='./data/train.json')
parser.add_argument('--samples_dir', default='./examples/example_images')
parser.add_argument('--samples_out', default=None)
parser.add_argument('--pretrain_model', default=None)

parser.add_argument('--save_loss_freq', type=int, default=20)
parser.add_argument('--save_model_freq', type=int, default=10000)
parser.add_argument('--log_dir', default='/home/chengkz/checkpoints/oppo_caption/log/{}')

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--fixed_len', type=int, default=20)
parser.add_argument('--lr_enc', type=float, default=2e-5)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--beam_num', type=int, default=5)
parser.add_argument('--beam_alpha', type=float, default=1.0)
parser.add_argument('--length_penalty', type=float, default=0.7)

parser.add_argument('--image_dim', type=int, default=2048)
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--att_dim', type=int, default=1024)

config = parser.parse_args()

