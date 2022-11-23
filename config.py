
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--GRID_SIZE', default=1000, type=int)
parser.add_argument('--max_grid', default=180, type=int)
parser.add_argument('--generate_num', default=500, type=int)
parser.add_argument('--generate_name', default=500, type=int)

# dir
parser.add_argument('--data_dir', default="../dataset/processed_info.json", type=str)
parser.add_argument('--split_data_dir', default="../dataset/split/", type=str)
parser.add_argument('--hw_dir', default="../dataset/user_hw.json", type=str)

# training
parser.add_argument('--max_pos', default=168, type=int, help="7*24, the max num of grids per day")
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--exp_name', type=str, default=None, help="experiment name")
parser.add_argument('--gpu', type=int, default=0, help="cuda id.")
parser.add_argument('--config', type=str, default='config_trans_ptb', help="The config to use.")
parser.add_argument('--mode', type=str, default='train', help="Train or predict. train/eval")
parser.add_argument('--model', type=str, default=None, help="Model path for generating sentences.")
parser.add_argument('--out', type=str, default=None, help="Generation output path.")
parser.add_argument('--model_name', type=str, default='pid')
parser.add_argument('--exp_kl', type=float, default=0, help="desired KL divergence.")
parser.add_argument('--Kp', type=float, default=0.01, help="Kp for pid.")
parser.add_argument('--Ki', type=float, default=-0.0001, help="Kp for pid.")
parser.add_argument('--cycle', type=float, default=4, help="Kp for pid.")
parser.add_argument('--max_steps', type=int, default=3000000, help="steps")
parser.add_argument('--num_epochs', default=500, type=int)

# model
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--embed_dim', default=128, type=int)
parser.add_argument('--latent_dims', default=64, type=int)
parser.add_argument('--init_lr', default=0.001, type=float)
parser.add_argument('--share_emb', default=False, action="store_true", help="share emb matrix between enc & dec emb")


args = parser.parse_args()