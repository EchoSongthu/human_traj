import importlib
import math
import os
import sys
import time
import numpy as np
import pickle
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
import texar.torch as tx
from texar.torch.custom import MultivariateNormalDiag
from PID import PIDControl
from tqdm import tqdm
from config import *
from utils import map_ids_to_tokens_py
from data_preprocess import myDataset
import pdb
import json
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f"../results/tensorboard_logs/exp_{args.exp_name}")


def main():

    config: Any = importlib.import_module(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    
    share_emb = args.share_emb
    if share_emb:
        print("share emb matrix")
        from model_share import VAE
    else:
        from model import VAE

    splits = ['train', 'val', 'test']
    dataset = {}
    dataloader = {}
    for i in splits:
        dataset[i] = myDataset(args, split=i)
        dataloader[i] = torch.utils.data.DataLoader(dataset=dataset[i], 
                                            batch_size=batch_size,
                                            shuffle=True)

    opt_vars = {
        'learning_rate': config.lr_decay_hparams["init_lr"],
        'best_valid_nll': 1e100,
        'steps_not_improved': 0,
        'kl_weight': config.kl_anneal_hparams["start"]
    }
    decay_cnt = 0
    max_decay = config.lr_decay_hparams["max_decay"]
    decay_factor = config.lr_decay_hparams["decay_factor"]
    decay_ts = config.lr_decay_hparams["threshold"]

    # initialize dir
    save_dir = '../results/results_KL' + str(args.exp_kl) + '_exp' + str(args.exp_name)# model save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    suffix = "model.ckpt"
    save_path = os.path.join(save_dir, suffix)

    # define model
    model = VAE(config)
    model.to(device)

    optimizer = tx.core.get_optimizer(params=model.parameters(),hparams=config.opt_hparams)
    scheduler = ExponentialLR(optimizer, decay_factor)

    # max iteration
    max_iter = int(config.num_epochs*len(dataset['train'])/args.batch_size)
    max_iter = min(max_iter, args.max_steps)
    print('max steps:', max_iter)
    pbar = tqdm(total = int(max_iter))
    
    if args.mode == "train":
        outFile = os.path.join(save_dir, 'train.log')
        fw_log = open(outFile, "w")
    
    global_steps = {}
    global_steps['step'] = 0
    pid = PIDControl()
    opt_vars["kl_weight"] = 0.0
    Kp = args.Kp
    Ki = args.Ki
    exp_kl = args.exp_kl

    ## train model
    def _run_epoch(epoch: int, mode: str, display: int = 10) -> Tuple[Tensor, float]:
        if mode == 'train':
            model.train()
            kl_weight = opt_vars["kl_weight"]
        else:
            model.eval()
            kl_weight = 1.0
        num_words = 0
        nll_total = 0.
        avg_rec = tx.utils.AverageRecorder()
        
        for batch in dataloader[mode]:
            
            ## run model to get loss function
            if global_steps['step']>= args.max_steps:
                break
            ret = model(batch, kl_weight)

            if mode == "train":
                pbar.update(1)
                global_steps['step'] += 1
                kl_loss = ret['kl_loss'].item()
                rec_loss = ret['rc_loss'].item()
                total_loss = ret["nll"].item()
                
                # use PID to update beta(kl_weight)
                if 'pid' in args.model_name: 
                    kl_weight = pid.pid(exp_kl, kl_loss, Kp, Ki) 
                
                opt_vars["kl_weight"] = kl_weight
                
                # total loss
                ret["nll"].backward()
                optimizer.step()
                optimizer.zero_grad()
                fw_log.write('epoch:{0} global_step:{1} total_loss:{2:.3f} kl_loss:{3:.3f} rec_loss:{4:.3f} kl_weight:{5:.4f}\n'\
                            .format(epoch, global_steps['step'], total_loss, kl_loss, rec_loss, kl_weight))
                fw_log.flush()

            batch_size = len(ret["lengths"])
            num_words += torch.sum(ret["lengths"]).item()

            nll_total += ret["nll"].item() * batch_size
            avg_rec.add(
                [ret["nll"].item(),
                 ret["kl_loss"].item(),
                 ret["rc_loss"].item()],
                batch_size)
                
            if global_steps['step'] % display == 1 and mode == 'train':
                nll = avg_rec.avg(0)
                klw = opt_vars["kl_weight"]
                KL = avg_rec.avg(1)
                rc = avg_rec.avg(2)
                writer.add_scalar(f'Loss/Rec_loss_{args.model_name}', rc, global_steps['step'])
                writer.add_scalar(f'Loss/KL_diverg_{args.model_name}', KL, global_steps['step'])
                writer.add_scalar(f'Loss/KL_weight_{args.model_name}', klw, global_steps['step'])
                
        nll = avg_rec.avg(0)
        KL = avg_rec.avg(1)
        rc = avg_rec.avg(2)
        if num_words > 0:
            log_ppl = nll_total / num_words
            ppl = math.exp(log_ppl)
        else:
            log_ppl = 100
            ppl = math.exp(log_ppl)
            nll = 1000
            KL = args.exp_kl
        
        print(f"\n{mode}: epoch {epoch}, nll {nll:.4f}, KL {KL:.4f}, "
              f"rc {rc:.4f}, log_ppl {log_ppl:.4f}, ppl {ppl:.4f}")
        return nll, ppl  # type: ignore
        
    # args.model = save_path

    # 生成
    @torch.no_grad()
    def _generate(start_tokens: torch.LongTensor,
                  end_token: int):
        model_path = os.path.join(args.model, suffix)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model'])
        model.eval()
        batch_size = args.batch_size

        latent_z = torch.FloatTensor(batch_size, config.latent_dims).uniform_(-1, 1).to(device)

        helper = model.decoder.create_helper(
            decoding_strategy='infer_sample',
            start_tokens=start_tokens,
            end_token=end_token)

        outputs = model.decode(
            helper=helper,
            latent_z=latent_z,
            max_decoding_length=args.max_pos)

        outputs = outputs[0]
        out_arr = outputs.sample_id.cpu().numpy()

        return out_arr
        # sample_tokens = map_ids_to_tokens_py(outputs.sample_id.cpu())

        # #写入文件（经纬度）
        # file_traj = []
        # for sent in sample_tokens:
        #     file_traj.append({
        #         "user_traj":sent}
        #     )
        # with open(filename, 'a') as f:
        #     json.dump(file_traj, f, indent=2, ensure_ascii=False)
        # print('Output done')


    # 生成
    if args.mode == "predict":
        generate_name = args.generate_name
        iter = args.generate_num//args.batch_size
        start_tokens = torch.full((args.batch_size,),0,dtype=torch.long).to(device)
        end_token = args.max_grid * args.max_grid + 1 #?
        # out_path = os.path.join(save_dir,'results.json')

        results = _generate(start_tokens, end_token)

        for i in range(iter):
            temp = _generate(start_tokens, end_token)
            results = np.concatenate((results,temp),axis=0)
    
        file_name = f"gen_data_epoch{generate_name}.pkl"
        path =os.path.join(args.model, file_name)
        pickle.dump(results, open(path,'wb'))
        return

    # Counts trainable parameters
    total_parameters = sum(param.numel() for param in model.parameters())
    print(f"{total_parameters} total parameters")
    best_nll = best_ppl = 0.

    ## start running model
    for epoch in range(config.num_epochs):
        _, _ = _run_epoch(epoch, 'train', display=200)
        val_nll, _ = _run_epoch(epoch, 'val')
        test_nll, test_ppl = _run_epoch(epoch, 'test')

        if val_nll < opt_vars['best_valid_nll']:
            opt_vars['best_valid_nll'] = val_nll
            opt_vars['steps_not_improved'] = 0
            best_nll = test_nll
            best_ppl = test_ppl

            states = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            torch.save(states, save_path)

        # 满 20 epoch 存一下
        if epoch % 20 == 0 and epoch != 0:
            temp_dir = os.path.join(save_dir, f"checkpoint_epoch{epoch}")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_path = os.path.join(temp_dir,suffix)
            torch.save(states, temp_path)

        else:
            opt_vars['steps_not_improved'] += 1
            if opt_vars['steps_not_improved'] == decay_ts:
                old_lr = opt_vars['learning_rate']
                opt_vars['learning_rate'] *= decay_factor
                opt_vars['steps_not_improved'] = 0
                new_lr = opt_vars['learning_rate']
                ckpt = torch.load(save_path)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                scheduler.step()
                print(f"-----\nchange lr, old lr: {old_lr}, "
                      f"new lr: {new_lr}\n-----")

                decay_cnt += 1
                if decay_cnt == max_decay:
                    break
        if global_steps['step'] >= args.max_steps:
            break

    print(f"\nbest testing nll: {best_nll:.4f},"
          f"best testing ppl {best_ppl:.4f}\n")
    
    if args.mode == "train":
        fw_log.write(f"\nbest testing nll: {best_nll:.4f},"
          f"best testing ppl {best_ppl:.4f}\n")
        fw_log.close()
        

if __name__ == '__main__':
    main()
    print("well done!!!!!")