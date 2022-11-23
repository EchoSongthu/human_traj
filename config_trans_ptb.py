from config import *
num_epochs = args.num_epochs
hidden_size = args.hidden_size
embed_dim = args.embed_dim
latent_dims = args.latent_dims
init_lr = args.init_lr

lr_decay_hparams = {
    'init_lr': init_lr,
    'threshold': 100,
    'decay_factor': 0.5,
    'max_decay': 5
}
relu_dropout = 0.2
embedding_dropout = 0.2
attention_dropout = 0.2
residual_dropout = 0.2
num_blocks = 3
max_pos = args.max_pos   # max sequence length in training data
enc_pos_emb_hparams = {'dim': hidden_size,}
dec_pos_emb_hparams = {'dim': hidden_size,}

# due to the residual connection, the embed_dim should be equal to hidden_size
trans_hparams = {
    'output_layer_bias': False,
    'embedding_dropout': embedding_dropout,
    'residual_dropout': residual_dropout,
    'num_blocks': num_blocks,
    'dim': hidden_size,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'factor': 1.0,
            'mode': 'FAN_AVG',
            'uniform': True,
        },
    },
    'multihead_attention': {
        'dropout_rate': attention_dropout,
        'num_heads': 8,
        'num_units': hidden_size,
        'output_dim': hidden_size
    },
    'poswise_feedforward': {
        'name': 'fnn',
        'layers': [
            {
                'type': 'Linear',
                'kwargs': {
                    "in_features": hidden_size,
                    "out_features": hidden_size * 4,
                    "bias": True,
                },
            },
            {
                'type': 'ReLU',
            },
            {
                'type': 'Dropout',
                'kwargs': {
                    'p': relu_dropout,
                }
            },
            {
                'type': 'Linear',
                'kwargs': {
                    "in_features": hidden_size * 4,
                    "out_features": hidden_size,
                    "bias": True,
                }
            }
        ],
    }
}

import texar
default_transformer_poswise_net_hparams = \
    texar.torch.modules.default_transformer_poswise_net_hparams(hidden_size,hidden_size)

trans_enc_hparams = {
    "num_blocks": 6,
    "dim": hidden_size,
    'use_bert_config': False,
    "embedding_dropout": 0.1,
    "residual_dropout": 0.1,
    "poswise_feedforward": default_transformer_poswise_net_hparams,
    'multihead_attention': {
        'name': 'multihead_attention',
        'num_units': 512,
        'num_heads': 8,
        'dropout_rate': 0.1,
        'output_dim': hidden_size,
        'use_bias': False,
    },
    "eps": 1e-6,
    "initializer": None,
    "name": "transformer_encoder"
}

kl_anneal_hparams = {
    'warm_up': 10,
    'start': 0.1
}

opt_hparams = {
    'optimizer': {
        'type': 'Adam',
        'kwargs': {
            'lr': 0.001
        }
    },
    'gradient_clip': {
        "type": "clip_grad_norm_",
        "kwargs": {
            "max_norm": 5,
            "norm_type": 2
        }
    }
}
