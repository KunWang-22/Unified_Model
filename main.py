import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import random
import json
import time
import os
import argparse
import tqdm

from moe import *
from data import *

import torchstat

def train(args, device):
    train_dataset = Dataset_UKDA(args.file_path, args.start_user, args.end_user)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    print(len(train_dataset))

    model = MoETransformer(seq_len=train_dataset[0].shape[0], num_patch=args.num_patch, num_layer=args.num_layer, d_in=args.input_dim, d_out=args.output_dim, d_patch=args.patch_dim, d_model=args.model_dim, d_hidden=args.hidden_dim, d_qkv=args.qkv_dim, num_head=args.num_head, num_expert=args.num_expert, dropout=args.dropout, top_k=args.top_k, training_flag=args.training_flag, mask_flag=args.mask_flag, mask_rate=args.mask_rate, importance_factor=args.importance_factor, load_factor=args.load_factor, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)

    # print(torchstat.stat(model, (8,48,1)))
    input()

    vib_loss_record = []
    aux_loss_record = []
    ixz_bound_record = []
    iyz_bound_record = []

    print("Start Traning !!!")
    model.train()
    for epoch in range(args.num_epoch):
        losses = []
        vib_loss_list = []
        aux_loss_list = []
        ixz_bound_list = []
        iyz_bound_list = []

        loop = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in loop:
            input_data = (data + torch.randn(data.shape)/2).permute(1,0,2).to(device)
            data = data.permute(1,0,2).to(device)
            # print(data.shape)
            # input()
            pred, aux_loss, encoder_mu, encoder_std, mask_data = model(input_data)

            info_loss = 0.5 * ((encoder_mu.pow(2) + encoder_std.pow(2) - 2*encoder_std.log() - 1)).sum(-1).sum(0).mean()
            mse_loss = criterion(data, pred)
            vib_loss = args.vib_beta*info_loss + mse_loss

            vib_loss_list.append(vib_loss.item())
            aux_loss_list.append(aux_loss.item())
            ixz_bound_list.append(info_loss.item())
            # remember to add "-" symbol in results !!!
            iyz_bound_list.append(mse_loss.item())

            # information bottleneck loss !
            loss = aux_loss + vib_loss
            # loss = aux_loss + mse_loss
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        print("Epoch {}/{}, Loss: {}, vib_loss: {}, aux_loss: {}, info_loss: {}, mse_loss: {}".format(epoch+1, args.num_epoch, np.array(losses, dtype=float).mean(), np.array(vib_loss_list, dtype=float).mean(), np.array(aux_loss_list, dtype=float).mean(), np.array(ixz_bound_list, dtype=float).mean(), np.array(iyz_bound_list, dtype=float).mean()))

        vib_loss_record.append(np.array(vib_loss_list).mean())
        aux_loss_record.append(np.array(aux_loss_list).mean())
        ixz_bound_record.append(np.array(ixz_bound_list).mean())
        iyz_bound_record.append(np.array(iyz_bound_list).mean())
        torch.save(model.state_dict(), "../log/{}/model.pt".format(args.dir_name))

    history_record = pd.DataFrame({
        "vib_loss": np.array(vib_loss_record),
        "aux_loss": np.array(aux_loss_record),
        "ixz_bound": np.array(ixz_bound_record),
        "iyz_bound": np.array(iyz_bound_record)
    })
    history_record.to_csv("../log/{}/history.csv".format(args.dir_name), index=None)



def eval(args, device):
    test_dataset = Dataset_UKDA(args.file_path, args.start_user, args.end_user)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    print(len(test_dataset))

    model = MoETransformer(seq_len=test_dataset[0].shape[0], num_patch=args.num_patch, num_layer=args.num_layer, d_in=args.input_dim, d_out=args.output_dim, d_patch=args.patch_dim, d_model=args.model_dim, d_hidden=args.hidden_dim, d_qkv=args.qkv_dim, num_head=args.num_head, num_expert=args.num_expert, dropout=args.dropout, top_k=args.top_k, training_flag=args.training_flag, mask_flag=args.mask_flag, mask_rate=args.mask_rate, importance_factor=args.importance_factor, load_factor=args.load_factor, device=device).to(device)
    if device == torch.device("cuda"):
        model.load_state_dict( torch.load("../log/{}/model.pt".format(args.dir_name)) )
    else:
        model.load_state_dict( torch.load("../log/{}/model.pt".format(args.dir_name), map_location=device) )


    print("Start Evaluating !!!")
    model.eval()
    real_data = []
    pred_data = []
    mask_data = []
    loop = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, data in loop:
        data = data.permute(1,0,2).to(device)
        pred, aux_loss, encoder_mu, encoder_std, mask = model(data)

        if args.mask_flag:
            mask = torch.cat(mask.cpu().split(1, dim=0), dim=-1).permute(2,1,0)

        if device == torch.device("cuda"):
            data = data.cpu()
            pred = pred.cpu()
            mask = mask.cpu()

        real_data.append(data.detach().permute(1,0,2).numpy().flatten())
        pred_data.append(pred.detach().permute(1,0,2).numpy().flatten())
        mask_data.append(mask.detach().permute(1,0,2).numpy().flatten())
    
    eval_data = pd.DataFrame({
        "real_data": np.array(real_data).flatten(),
        "pred_data": np.array(pred_data).flatten(),
        "mask_data": np.array(mask_data).flatten()
    })

    eval_data.to_csv("../log/{}/result.csv".format(args.dir_name), index=None)



def random_setup(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)



def save_args(args):
    with open('../log/{}/args.txt'.format(args.dir_name), 'w') as f:
        json.dump(args.__dict__, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("The hyper-parameters of this project")
    # # Dataset
    parser.add_argument("--file_path", type=str, default="../data/UKDA_2013_clean.csv")
    parser.add_argument("--start_user", type=int, default=0)
    parser.add_argument("--end_user", type=int, default=20)

    # Model
    parser.add_argument("--num_layer", type=int, default=8)
    parser.add_argument("--num_expert", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)

    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--patch_dim", type=int, default=4)
    parser.add_argument("--num_patch", type=int, default=12)
    
    parser.add_argument("--model_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--qkv_dim", type=int, default=4)
    parser.add_argument("--num_head", type=int, default=4)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--training_flag", action='store_true', default=False)
    parser.add_argument("--mask_flag", action='store_true', default=False)
    parser.add_argument("--mask_rate", type=float, default=0.2)

    parser.add_argument("--importance_factor", type=float, default=0.1)
    parser.add_argument("--load_factor", type=float, default=0.1)
    parser.add_argument("--vib_beta", type=float, default=0.001)

    # Train
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda', action='store_true', default=False)

    parser.add_argument("--dir_name", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=22)
    parser.add_argument("--run_time", type=float, default=0.0)

    # Device
    args = parser.parse_args()
    args.cuda = not args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dir_name == None:
        args.dir_name = time.strftime("%Y_%m_%d_%H_%M")
        os.makedirs("../log/{}".format(args.dir_name))

    random_setup(args.random_seed)
    print(args.training_flag)
    if args.training_flag:
        time_start = time.time()
        train(args, device)
        time_end = time.time()
        args.run_time = time_end - time_start
        save_args(args)
    else:
        eval(args, device)