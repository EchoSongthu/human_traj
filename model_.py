import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from config import *
torch.set_printoptions(threshold=np.inf)
torch.cuda.set_device(args.gpu)

# torch.set_default_tensor_type(torch.FloatTensor)
from typing import Any, Dict, Optional, Tuple, Union
import texar.torch as tx
from texar.torch.custom import MultivariateNormalDiag
import pdb


def kl_divergence(means: Tensor, logvars: Tensor) -> Tensor:
    """Compute the KL divergence between Gaussian distribution
    """
    kl_cost = -0.5 * (logvars - means ** 2 -
                      torch.exp(logvars) + 1.0)
    kl_cost = torch.mean(kl_cost, 0)
    return torch.sum(kl_cost)


class VAE(nn.Module):
    _latent_z: Tensor

    def __init__(self, config_model):
        super().__init__()
        # Model architecture
        self._config = config_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_grid = args.max_grid
        self.num_grid = self.max_grid * self.max_grid +1
        self.max_week = 8
        self.max_hour = 25
        self.max_pos = args.max_pos

        self.generate_start_week = 0 #用于指定generate的起始week
        self.generate_start_hour = 0

        self.sum_state_size = args.embed_dim

        self.encoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
                position_size=config_model.max_pos+1,
                hparams=config_model.enc_pos_emb_hparams)
        self.encoder_traj_embedder = nn.Linear(self.num_grid, self.sum_state_size)
        self.encoder_week_embedder = nn.Linear(self.max_week, self.sum_state_size)
        self.encoder_hour_embedder = nn.Linear(self.max_hour, self.sum_state_size)

        self.transformer_encoder = tx.modules.TransformerEncoder(config_model.trans_enc_hparams)
        
        self.decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
                position_size=config_model.max_pos,
                hparams=config_model.dec_pos_emb_hparams)
        self.decoder_traj_embedder = nn.Linear(self.num_grid, self.sum_state_size)
        self.decoder_week_embedder = nn.Linear(self.max_week, self.sum_state_size)
        self.decoder_hour_embedder = nn.Linear(self.max_hour, self.sum_state_size)
        self.transformer_decoder = tx.modules.TransformerDecoder(
                # tie word embedding with output layer
                output_layer=self.decoder_traj_embedder.weight.t(),
                token_pos_embedder=self._embed_fn_transformer,
                hparams=config_model.trans_hparams)

        self.connector_mlp = tx.modules.MLPTransformConnector(
                                    config_model.latent_dims *2,
                                    linear_layer_dim=self.transformer_encoder._input_size)
        self.mlp_linear_layer = nn.Linear(config_model.latent_dims, self.sum_state_size)

        
    def forward(self, data_batch, kl_weight: float) -> Dict[str, Tensor]: # bs,max_pos,4
        data_batch_ = data_batch["data"]
        bs = data_batch_.shape[0]

        input_traj = data_batch_[:,:,0,...].unsqueeze(2)
        input_traj_ = torch.zeros((input_traj.shape[0],input_traj.shape[1], self.num_grid), device=self.device)\
                            .scatter_(2,input_traj,1)

        input_hour = data_batch_[:,:,1,...].unsqueeze(2)
        input_hour_ = torch.zeros((input_hour.shape[0],input_hour.shape[1], self.max_hour), device=self.device)\
                            .scatter_(2,input_hour,1)
        input_week = data_batch_[:,:,2,...].unsqueeze(2)
        input_week_ = torch.zeros((input_week.shape[0],input_week.shape[1], self.max_week), device=self.device)\
                            .scatter_(2,input_week,1)

        positions = torch.stack([torch.arange(0,self.max_pos+1,1) for i in range(bs)],dim=0)
        positions = positions.to(self.device) # bs,673
        
        input_traj_embed = self.encoder_traj_embedder(input_traj_.float())
        input_hour_embed = self.encoder_hour_embedder(input_hour_.float())
        input_week_embed = self.encoder_week_embedder(input_week_.float())
        input_traj_embed = input_traj_embed * self._config.hidden_size ** 0.5
        input_p_embed = self.encoder_p_embedder(positions)
        
        input_embed = input_traj_embed + input_hour_embed + input_week_embed + input_p_embed  # 32,169,256
        # pdb.set_trace()

        encoder_states = self.transformer_encoder(
                        input_embed,
                        sequence_length=data_batch["length"])
        encoder_states = encoder_states[:,0,:]

        mean_logvar = self.connector_mlp(encoder_states)
        mean, logvar = torch.chunk(mean_logvar, 2, 1)

        kl_loss = kl_divergence(mean, logvar)
        dst = MultivariateNormalDiag(loc=mean, scale_diag=torch.exp(0.5 * logvar))
        latent_z = dst.rsample()

        helper = None
        seq_lengths = data_batch["length"].to(self.device) - 1
        outputs = self.decode(
                    helper=helper, latent_z=latent_z,
                    text_ids = data_batch_[:, :-1], seq_lengths=seq_lengths)
        logits = outputs.logits

        rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                    labels=data_batch_[:,:,0,...][:, 1:], logits=logits,
                    sequence_length=seq_lengths)
        total_loss = rc_loss + kl_weight * kl_loss
        ret = {
            "nll": total_loss,
            "kl_loss": kl_loss,
            "rc_loss": rc_loss,
            "lengths": seq_lengths,
        }
        return ret


    
    def _embed_fn_transformer(self,
                              tokens: torch.LongTensor,
                              positions: torch.LongTensor)-> Tensor:
        r"""Generates word embeddings combined with positional embeddings
        """
        if tokens.shape != (args.batch_size,):
            # train
            output_traj = tokens[:,:,0,...].unsqueeze(2)
            output_hour = tokens[:,:,1,...].unsqueeze(2)
            output_week = tokens[:,:,2,...].unsqueeze(2)
            output_traj_ = torch.nn.Parameter(torch.zeros((output_traj.shape[0],output_traj.shape[1], self.num_grid), device=self.device)\
                                .scatter_(2,output_traj,1), requires_grad=False)
            output_hour_ = torch.zeros((output_hour.shape[0],output_hour.shape[1], self.max_hour), device=self.device)\
                                .scatter_(2,output_hour,1)
            output_week_ = torch.zeros((output_week.shape[0],output_week.shape[1], self.max_week), device=self.device)\
                                .scatter_(2,output_week,1)
            
        else:# generate
            if positions[0].item()==0:
                self.generate_start_week = np.random.randint(low=1,high=8) # 序列开始，随机产生一个起始星期几
                hour = 0
                week = 0
            else:
                week = ( (positions[0].item()-1)//24 + self.generate_start_week) % 8
                if week ==0:
                    week += 1
                    self.generate_start_week += 1

                hour = (positions[0].item() + self.generate_start_hour) % 25
                if hour == 0:
                    hour += 1
                    self.generate_start_hour += 1

            output_traj = tokens.unsqueeze(1)
            output_hour = torch.full((args.batch_size,),hour,dtype=torch.long).to(self.device).unsqueeze(1)
            output_week = torch.full((args.batch_size,),week,dtype=torch.long).to(self.device).unsqueeze(1)

            # print(self.generate_start_week, output_traj[0].item(), output_hour[0].item(), output_week[0].item())
            output_traj_ = torch.zeros((output_traj.shape[0], self.num_grid), device=self.device)\
                                .scatter_(1,output_traj,1)
            output_hour_ = torch.zeros((output_hour.shape[0], self.max_hour), device=self.device)\
                                .scatter_(1,output_hour,1)
            output_week_ = torch.zeros((output_week.shape[0], self.max_week), device=self.device)\
                                .scatter_(1,output_week,1)
            
        output_traj_embed = self.decoder_traj_embedder(output_traj_.float())
        output_hour_embed = self.decoder_hour_embedder(output_hour_.float())
        output_week_embed = self.decoder_week_embedder(output_week_.float())
        output_p_embed = self.decoder_p_embedder(positions)

        output_traj_embed = output_traj_embed * self._config.hidden_size ** 0.5
        output_embed = output_traj_embed + output_hour_embed + output_week_embed + output_p_embed 

        return output_embed

    @property
    def decoder(self) -> tx.modules.DecoderBase:
        return self.transformer_decoder

    def decode(self,
               helper: Optional[tx.modules.Helper],
               latent_z: Tensor,
               text_ids: Optional[torch.LongTensor] = None,
               seq_lengths: Optional[Tensor] = None,
               max_decoding_length: Optional[int] = None) \
            -> Union[tx.modules.BasicRNNDecoderOutput,
                     tx.modules.TransformerDecoderOutput]:

        fc_output = self.mlp_linear_layer(latent_z)
        transformer_states = fc_output.unsqueeze(1) # 4,1,256

        outputs = self.transformer_decoder(
            inputs=text_ids,
            memory=transformer_states,
            memory_sequence_length=torch.ones(transformer_states.size(0)),
            helper=helper,
            max_decoding_length=max_decoding_length)

        return outputs
