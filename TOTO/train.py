#!/usr/bin/python3

# ==============================================================================
# -- COPYRIGHT -----------------------------------------------------------------
# ==============================================================================

# Copyright (c) 2022 SLED Lab, EECS, University of Michigan.
# authors: Martin Ziqiao Ma (marstin@umich.edu),
#          Cristian-Paul Bara (cpbara@umich.edu),
#          Owen Yidong Huang (owenhji@umich.edu),
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

import json
import os, sys
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

from nn.action_encoder import ActionEncoder
# from nn.knowledge_encoder import LaneEncoder
from ontology.agent import PhysicalAction, DialogueMove, DialogueSlot
from ontology.geometry import Lane

from data.dataset import Game

from model.toto_plan import TOTO
import torch
from torch import optim, nn
import math
from multiprocessing import Pool, Manager

from tqdm import tqdm
from functools import partial
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from copy import deepcopy

from random import shuffle
import gc
from toto_dataset import Toto_Dataset
import wandb

from transformers import BertTokenizer, BertModel
format_time = lambda x: f'{int(x)//3600:02d}:{int(x)%3600//60:02d}:{int(x%60):02d}'

def do_epoch(args, model, optimizer, criterion_cel, criterion_mse,criterion_bce, events, do_eval=False):
    loss_lst = []
    own_pos_err = []
    own_angle_err = []
    goal_pos_err = []
    act_type_pairs = []
    act_angle_err = []
    dorothy_dlg_move_pairs = []
    dorothy_type_rec = []
    dorothy_slot_rec = []
    wizzard_dlg_move_pairs = []
    wizzard_type_rec = []
    wizzard_slot_rec = []

    belief_loss_list=[]

    physical_type_loss_list=[]

    physical_angle_loss_list=[]

    dorothy_dialogue_move_loss_list=[]

    dorothy_dialogue_slot_type_loss_list=[]
    dorothy_dialogue_slot_content_loss_list=[]
    
    wizard_dialogue_move_loss_list=[]

    wizard_dialogue_slot_type_loss_list=[]
    wizard_dialogue_slot_content_loss_list=[]
    # num_epochs = 1
    for event in tqdm(events):
                # print(event["detailed_plan"])
                lanes = event["lanes"]
                while len(lanes) < model.knowledge_encoder.max_num_nodes:
                    lanes.append(Lane(None))
                knowledge_emb = model.knowledge_encoder(
                    lanes, 
                    torch.tensor(event["edge_index"]).t().long().to(args.device, non_blocking=True), 
                    event["landmarks"],
                    event["landmark_name"].to(args.device, non_blocking=True),
                    event["streets"],
                    event["street_name"].to(args.device, non_blocking=True)
                ).float()
                idx = event['frame']
                obj_def_detr=event["obj_def_detr"].to(args.device, non_blocking=True).float()
                rgb_emb=event["rgb"].to(args.device, non_blocking=True).float()
                speech_emb=event["speech"].to(args.device, non_blocking=True).float()
                belief=event["belief"]
                if len(event["dialog_history_idx"])!=0:
                    dialog_history=event["dialog_history"].to(args.device, non_blocking=True).float()
                    dialog_history_dir=event["dialog_history_dir"].to(args.device, non_blocking=True).float()
                    dialog_history_move=model.dlg_move_type_encoder(event["dialog_history_move"].to(args.device, non_blocking=True)).float()
                    dialog_history_slot=event["dialog_history_slot"].to(args.device, non_blocking=True).float()
                    dialog_history_type=event["dialog_history_type"].to(args.device, non_blocking=True).float()
                    dlg_history=dialog_history
                    dlg_move_history=torch.cat([dialog_history_dir,dialog_history_move,dialog_history_type,dialog_history_slot],dim=1)
                    
                else:
                    dlg_history=torch.zeros([0,770]).to(args.device, non_blocking=True).float()
                    dlg_move_history=torch.zeros([0,770]).to(args.device, non_blocking=True).float()

                act_history_type=model.action_encoder(event["act_history_type"].to(args.device, non_blocking=True)).float()
                act_history_val=event["act_history_val"].to(args.device, non_blocking=True).float()
                act_history=torch.cat([act_history_type,act_history_val.unsqueeze(1)],dim=1)
                belief_x=torch.tensor([traj['x']for traj in belief]).to(args.device, non_blocking=True).unsqueeze(1).float()
                belief_y=torch.tensor([traj['y']for traj in belief]).to(args.device, non_blocking=True).unsqueeze(1).float()
                belief_yaw=torch.tensor([traj['yaw']for traj in belief]).to(args.device, non_blocking=True).unsqueeze(1).float()
                belief_lane=torch.tensor([traj['lane_id']for traj in belief]).to(args.device, non_blocking=True).unsqueeze(1).float()
                belief_road=torch.tensor([traj['road_id'] if traj['road_id']!=-1 else 0 for traj in belief]).int().to(args.device, non_blocking=True)
                belief_junc=torch.tensor([traj['junction_id'] if traj['junction_id']!=-1 else 0 for traj in belief]).int().to(args.device, non_blocking=True)
                belief_emb=torch.cat([belief_x,belief_y,belief_yaw,belief_lane],dim=1)
                # knowledge_emb=torch.zeros_like(knowledge_emb).to(args.device, non_blocking=True)
                outputs = model(obj_def_detr,rgb_emb,knowledge_emb,speech_emb,dlg_history,dlg_move_history,act_history.detach(),belief_emb,belief_road,belief_junc ,event)
            
                belief , physical_type , physical_angle , dorothy_dialogue_move , dorothy_dialogue_type ,dorothy_dialogue_slot , wizzard_dialogue_move , wizzard_dialogue_type, wizzard_dialogue_slot  = outputs
                





                gt_belief=event["gt_belief"].to(args.device, non_blocking=True).unsqueeze(0)
                if event["gt_physical_type"]!=None:
                    gt_physical_type=event["gt_physical_type"].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_physical_type=None


                if event["gt_physical_angle"]!=None:
                    gt_physical_angle =event["gt_physical_angle"].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_physical_angle=None

                if event['gt_dorothy_dialogue_move']!=None:
                    gt_dorothy_dialogue_move =event['gt_dorothy_dialogue_move'].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_dorothy_dialogue_move=None


                    
                if event['gt_dorothy_dialogue_type']!=None:
                    gt_dorothy_dialogue_type =event['gt_dorothy_dialogue_type'].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_dorothy_dialogue_type=None


                if event['gt_dorothy_dialogue_slot']!=None:
                    gt_dorothy_dialogue_slot =event['gt_dorothy_dialogue_slot'].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_dorothy_dialogue_slot=None

                if event['gt_wizzard_dialogue_move']!=None:
                    gt_wizzard_dialogue_move =event['gt_wizzard_dialogue_move'].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_wizzard_dialogue_move=None

                if event['gt_wizzard_dialogue_type']!=None:
                    gt_wizzard_dialogue_type =event['gt_wizzard_dialogue_type'].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_wizzard_dialogue_type=None

                if event['gt_wizzard_dialogue_slot']!=None:
                    gt_wizzard_dialogue_slot =event['gt_wizzard_dialogue_slot'].to(args.device, non_blocking=True).unsqueeze(0)
                else:
                    gt_wizzard_dialogue_slot=None

                
                loss=0
                belief_loss = 1e-3*torch.sqrt(criterion_mse(belief ,gt_belief )) * int(args.belief_out)
                belief_loss_list.append(belief_loss.item())
                loss+=belief_loss

                if gt_physical_type !=None:
                    physical_type_loss = criterion_cel(physical_type .float(), gt_physical_type )
                    loss+=physical_type_loss
                    physical_type_loss_list.append(physical_type_loss.item())

                if gt_physical_angle !=None:
                    physical_angle_loss = wandb.config["angle_loss_weight"]*torch.sqrt(criterion_mse(physical_angle , gt_physical_angle .float().reshape(-1,1)))
                    loss+=physical_angle_loss
                    physical_angle_loss_list.append(physical_angle_loss.item())
                
                if gt_dorothy_dialogue_move !=None:
                    dorothy_dialogue_move_loss = criterion_cel(dorothy_dialogue_move .float(), gt_dorothy_dialogue_move )
                    loss+=dorothy_dialogue_move_loss
                    dorothy_dialogue_move_loss_list.append(dorothy_dialogue_move_loss.item())
                
                if gt_dorothy_dialogue_type !=None:
                    dorothy_dialogue_slot_type_loss = wandb.config["type_loss_weight"]*criterion_bce(dorothy_dialogue_type.float(), gt_dorothy_dialogue_type.float() )
                    dorothy_dialogue_slot_content_loss = wandb.config["content_loss_weight"]*criterion_bce(dorothy_dialogue_slot.float(), gt_dorothy_dialogue_slot.float() )
                    loss+=dorothy_dialogue_slot_type_loss+dorothy_dialogue_slot_content_loss
                    dorothy_dialogue_slot_type_loss_list.append(dorothy_dialogue_slot_type_loss.item())
                    dorothy_dialogue_slot_content_loss_list.append(dorothy_dialogue_slot_content_loss.item())
                    
                if gt_wizzard_dialogue_move !=None:
                    # print(gt_wizzard_dialogue_move)
                    # print(wizzard_dialogue_move.shape)
                    wizard_dialogue_move_loss = criterion_cel(wizzard_dialogue_move.float(), gt_wizzard_dialogue_move)
                    loss+=wizard_dialogue_move_loss
                    wizard_dialogue_move_loss_list.append(wizard_dialogue_move_loss.item())
                
                if gt_wizzard_dialogue_type !=None:
                    wizard_dialogue_slot_type_loss = wandb.config["type_loss_weight"]*criterion_bce(wizzard_dialogue_type .float(), gt_wizzard_dialogue_type.float()  ) 
                    wizard_dialogue_slot_content_loss =  wandb.config["content_loss_weight"]*criterion_bce(wizzard_dialogue_slot.float(), gt_wizzard_dialogue_slot.float() )
                    loss+=wizard_dialogue_slot_type_loss+wizard_dialogue_slot_content_loss
                    wizard_dialogue_slot_type_loss_list.append(wizard_dialogue_slot_type_loss.item())
                    wizard_dialogue_slot_content_loss_list.append(wizard_dialogue_slot_content_loss.item())
                    
                # print(loss)
                if not do_eval:    
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                loss_lst.append(loss.item())


                slot_eval_fun = lambda x,y: [f1_score(a,b,average='weighted',zero_division=1) for a,b in zip(x,y)]
                belief_sqr_err = (belief-gt_belief)**2
                own_pos_err += list(torch.sqrt(belief_sqr_err[:,0]+belief_sqr_err[:,1]).cpu().data.numpy())
                own_angle_err += list(torch.sqrt(belief_sqr_err[:,2]%180).cpu().data.numpy())
                goal_pos_err += list(torch.sqrt(belief_sqr_err[:,3]+belief_sqr_err[:,4]).cpu().data.numpy())
        

                if gt_physical_type !=None:
                    act_type_pairs += list(zip(torch.argmax(physical_type,axis=-1).cpu().data.numpy(),gt_physical_type.cpu().data.numpy()))

                if gt_physical_angle !=None:
                    act_angle_err += list((torch.abs(physical_angle-gt_physical_angle)<15).cpu().data.numpy())
                
                if gt_dorothy_dialogue_move !=None:
                    dorothy_dlg_move_pairs += list(zip(torch.argmax(dorothy_dialogue_move,axis=-1).cpu().data.numpy(),gt_dorothy_dialogue_move.cpu().data.numpy()))
                    # print(dorothy_dialogue_type.shape,torch.argmax(dorothy_dialogue_type, axis=-1),gt_dorothy_dialogue_type)
                if gt_dorothy_dialogue_type !=None:
                    dorothy_type_rec += slot_eval_fun((dorothy_dialogue_type>0).cpu().data.numpy(),gt_dorothy_dialogue_type.cpu().data.numpy())
                    if gt_dorothy_dialogue_type[0,0]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[0:10]>0).cpu().data.numpy(), gt_dorothy_dialogue_slot[0:10].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,1]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[10:26]>0).cpu().data.numpy(), gt_dorothy_dialogue_slot[10:26].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,2]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[26:32]>0).cpu().data.numpy(), gt_dorothy_dialogue_slot[26:32].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,3]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[32:44]>0).cpu().data.numpy(), gt_dorothy_dialogue_slot[32:44].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,4]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[44:]>0).cpu().data.numpy(), gt_dorothy_dialogue_slot[44:].cpu().data.numpy() )
                    
                if gt_wizzard_dialogue_move !=None:
                    wizzard_dlg_move_pairs += list(zip(torch.argmax(wizzard_dialogue_move,axis=-1).cpu().data.numpy(),gt_wizzard_dialogue_move.cpu().data.numpy()))
                if gt_wizzard_dialogue_type !=None:
                    wizzard_type_rec += slot_eval_fun((wizzard_dialogue_type>0).cpu().data.numpy(),gt_wizzard_dialogue_type.cpu().data.numpy())
                    if gt_wizzard_dialogue_type[0,0]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[0:10]>0).cpu().data.numpy(), gt_wizzard_dialogue_slot[0:10].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,1]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[10:26]>0).cpu().data.numpy(), gt_wizzard_dialogue_slot[10:26].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,2]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[26:32]>0).cpu().data.numpy(), gt_wizzard_dialogue_slot[26:32].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,3]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[32:44]>0).cpu().data.numpy(), gt_wizzard_dialogue_slot[32:44].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,4]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[44:]>0).cpu().data.numpy(), gt_wizzard_dialogue_slot[44:].cpu().data.numpy() )
                    

        
    

        
    return loss_lst, own_pos_err, own_angle_err, goal_pos_err, act_type_pairs, act_angle_err, dorothy_dlg_move_pairs, dorothy_type_rec,dorothy_slot_rec, wizzard_dlg_move_pairs, wizzard_type_rec,wizzard_slot_rec,belief_loss_list,physical_type_loss_list,physical_angle_loss_list,dorothy_dialogue_move_loss_list,dorothy_dialogue_slot_type_loss_list,dorothy_dialogue_slot_content_loss_list,wizard_dialogue_move_loss_list,wizard_dialogue_slot_type_loss_list,wizard_dialogue_slot_content_loss_list

def main(proc_idx, train_events, val_events, args):

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    run_pth = f'ablation{proc_idx}_seed{args.seed}_{int(time.time())}'
    os.mkdir(os.path.join(args.out_path,run_pth))
    file_name = os.path.join(args.out_path,run_pth,'train_log.log')
    out_file = open(file_name,'w')
    close_file = True


    # if proc_idx%2:
    #     # train_games = train_games[0::2] + train_games[1::2]
    #     # val_games = val_games[0::2] + val_games[1::2]
    #     # train_games = sum(([train_games[0:3]+train_games[0:2]])*(len(train_games)//5),[])
    #     # val_games = sum(([val_games[0:3]+val_games[0:2]])*(len(val_games)//5),[])
    #     pass
    # else:
    #     # train_games = train_games[0::3] + train_games[1::3] + train_games[2::3]
    #     # val_games = val_games[0::3] + val_games[1::3] + val_games[2::3]
    #     # train_games = sum(([train_games[2:5]+train_games[3:5]])*(len(train_games)//5),[])
    #     # val_games = sum(([val_games[3:5]+val_games[3:5]])*(len(val_games)//5),[])
    #     
    if proc_idx == 1:
        pass
    else:
        args.belief_out = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(args, file=out_file)

    
    # print(args.dataset_path)
    # print(game_paths)

    model = TOTO(args).to(args.device, non_blocking=True)
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num params', pytorch_total_params, file=out_file, flush=True)
    print('Model Loaded', file=out_file, flush=True)

    # print(game_paths)
    # exit()
    # games = [Game(game_path,args) for game_path in game_paths]


    learning_rate = args.lr
    num_epochs = 1000#2#1#
    weight_decay= args.weight_decay

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_cel = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    times = []
    # loss_lst = []
    # own_pos_err = []
    # own_angle_err = []
    # goal_pos_err = []
    # act_type_pairs = []
    # act_angle_err = []
    # dorothy_dlg_type_pairs = []
    # dorothy_slot_rec = []
    # wizzard_dlg_type_pairs = []
    # wizzard_slot_rec = []

    max_num_epochs = 25#30
    min_num_epochs = 20#15
    start = time.time()
    prev_loss = 1e10
    epochs_since_improvement = 0
    max_wait_time = 5
    print('                                   loss                belief                    Act                   dlg Dorothy             dlg Wizzard', file=out_file, flush=True)
    print('                              avg      stdev  own err    angle  gaol err    type   angle      move       type     slot    move     type     slot', file=out_file, flush=True)



    for epoch in range(max_num_epochs):
        model.train()
        shuffle(train_events)

        output = do_epoch(args, model, optimizer, criterion_cel, criterion_mse,criterion_bce, train_events)
        print("one_train")
        loss_lst, own_pos_err, own_angle_err, goal_pos_err, act_type_pairs, act_angle_err, dorothy_dlg_move_pairs, dorothy_type_rec,dorothy_slot_rec, wizzard_dlg_move_pairs, wizzard_type_rec,wizzard_slot_rec,belief_loss_list,physical_type_loss_list,physical_angle_loss_list,dorothy_dialogue_move_loss_list,dorothy_dialogue_slot_type_loss_list,dorothy_dialogue_slot_content_loss_list,wizard_dialogue_move_loss_list,wizard_dialogue_slot_type_loss_list,wizard_dialogue_slot_content_loss_list = output
        # for o in output:
        #     print(len(o))


        # print(act_type_pairs)
        # print(list(zip(*act_type_pairs)))
        # print(accuracy_score(*zip(*act_type_pairs)))
        # exit()
        print(
            f'Epoch {epoch+1:4d}',
            '               ',
            f'{np.mean(loss_lst):8.3f}',
            f'{np.std(loss_lst):8.3f}',
            f'{np.mean(own_pos_err):8.3f}',
            f'{np.mean(own_angle_err):8.3f}',
            f'{np.mean(goal_pos_err):8.3f}',
            f'{accuracy_score(*zip(*act_type_pairs)):8.3f}',
            f'{np.mean(act_angle_err):8.3f}',
            f'{accuracy_score(*zip(*dorothy_dlg_move_pairs)):8.3f}',
            f'{np.mean(dorothy_type_rec):8.3f}',
            f'{np.mean(dorothy_slot_rec):8.3f}',
            f'{accuracy_score(*zip(*wizzard_dlg_move_pairs)):8.3f}',
            f'{np.mean(wizzard_type_rec):8.3f}',
            f'{np.mean(wizzard_slot_rec):8.3f}',
            'Training',
            file=out_file,
            flush=True
            )
        train_loss = np.mean(loss_lst)
        train_physic_move = accuracy_score(*zip(*act_type_pairs))
        train_dorothy_dlg= accuracy_score(*zip(*dorothy_dlg_move_pairs))
        train_wizard_dlg = accuracy_score(*zip(*wizzard_dlg_move_pairs))
        train_belief_loss = np.mean(belief_loss_list)
        train_physical_type_loss = np.mean(physical_type_loss_list)
        train_hysical_angle_loss = np.mean(physical_angle_loss_list)
        train_dorothy_dialogue_move_loss = np.mean(dorothy_dialogue_move_loss_list)
        train_dorothy_dialogue_slot_type_loss = np.mean(dorothy_dialogue_slot_type_loss_list)
        
        train_dorothy_dialogue_slot_content_loss = np.mean(dorothy_dialogue_slot_content_loss_list)
        train_wizard_dialogue_move_loss=np.mean(wizard_dialogue_move_loss_list)
        train_wizard_dialogue_slot_type_loss=np.mean(wizard_dialogue_slot_type_loss_list)
        train_wizard_dialogue_slot_content_loss=np.mean(wizard_dialogue_slot_content_loss_list)
        
        model.eval()

        output = do_epoch(args, model, optimizer, criterion_cel, criterion_mse,criterion_bce, val_events, do_eval=True)
        loss_lst, own_pos_err, own_angle_err, goal_pos_err, act_type_pairs, act_angle_err, dorothy_dlg_move_pairs, dorothy_type_rec,dorothy_slot_rec, wizzard_dlg_move_pairs, wizzard_type_rec,wizzard_slot_rec,belief_loss_list,physical_type_loss_list,physical_angle_loss_list,dorothy_dialogue_move_loss_list,dorothy_dialogue_slot_type_loss_list,dorothy_dialogue_slot_content_loss_list,wizard_dialogue_move_loss_list,wizard_dialogue_slot_type_loss_list,wizard_dialogue_slot_content_loss_list = output
        # for o in output:
        #     print(len(o))


        times.append(time.time()-start)

        print(
            format_time(sum(times)), 
            format_time(np.mean(times)*(max_num_epochs-epoch-1)),
            format_time(sum(times)+np.mean(times)*(max_num_epochs-epoch-1)),
            f'{np.mean(loss_lst):8.3f}',
            f'{np.std(loss_lst):8.3f}',
            f'{np.mean(own_pos_err):8.3f}',
            f'{np.mean(own_angle_err):8.3f}',
            f'{np.mean(goal_pos_err):8.3f}',
            f'{accuracy_score(*zip(*act_type_pairs)):8.3f}',
            f'{np.mean(act_angle_err):8.3f}',
            f'{accuracy_score(*zip(*dorothy_dlg_move_pairs)):8.3f}',
            f'{np.mean(dorothy_type_rec):8.3f}',
            f'{np.mean(dorothy_slot_rec):8.3f}',
            f'{accuracy_score(*zip(*wizzard_dlg_move_pairs)):8.3f}',
            f'{np.mean(wizzard_type_rec):8.3f}',
            f'{np.mean(wizzard_slot_rec):8.3f}',
            'Evaluation',
            end = ' ',
            file=out_file,
            flush=True
            )
        val_loss = np.mean(loss_lst)
        val_physic_move = accuracy_score(*zip(*act_type_pairs))
        val_dorothy_dlg= accuracy_score(*zip(*dorothy_dlg_move_pairs))
        val_wizard_dlg = accuracy_score(*zip(*wizzard_dlg_move_pairs))
        val_belief_loss = np.mean(belief_loss_list)
        val_physical_type_loss = np.mean(physical_type_loss_list)
        val_hysical_angle_loss = np.mean(physical_angle_loss_list)
        val_dorothy_dialogue_move_loss = np.mean(dorothy_dialogue_move_loss_list)
        val_dorothy_dialogue_slot_type_loss = np.mean(dorothy_dialogue_slot_type_loss_list)
        
        val_dorothy_dialogue_slot_content_loss = np.mean(dorothy_dialogue_slot_content_loss_list)
        val_wizard_dialogue_move_loss=np.mean(wizard_dialogue_move_loss_list)
        val_wizard_dialogue_slot_type_loss=np.mean(wizard_dialogue_slot_type_loss_list)
        val_wizard_dialogue_slot_content_loss=np.mean(wizard_dialogue_slot_content_loss_list)
        
        wandb.log({
            "epoch": epoch,
            "train_loss" : train_loss,
            "train_physic_move": train_physic_move,
            "train_dorothy_dlg": train_dorothy_dlg,
            "train_wizard_dlg": train_wizard_dlg,
            "train_belief_loss" : train_belief_loss,
            "train_physical_type_loss" : train_physical_type_loss,
            "train_hysical_angle_loss" : train_hysical_angle_loss,
            "train_dorothy_dialogue_move_loss" : train_dorothy_dialogue_move_loss,
            "train_dorothy_dialogue_slot_type_loss" : train_dorothy_dialogue_slot_type_loss,
            
            "train_dorothy_dialogue_slot_content_loss" : train_dorothy_dialogue_slot_content_loss,
            "train_wizard_dialogue_move_loss":train_wizard_dialogue_move_loss,
            "train_wizard_dialogue_slot_type_loss":train_wizard_dialogue_slot_type_loss,
            "train_wizard_dialogue_slot_content_loss":train_wizard_dialogue_slot_content_loss,
            "val_loss":val_loss,
            "val_physic_move":val_physic_move,
            "val_dorothy_dlg": val_dorothy_dlg,
            "val_wizard_dlg": val_wizard_dlg,
            "val_belief_loss" : val_belief_loss,
            "val_physical_type_loss" : val_physical_type_loss,
            "val_hysical_angle_loss" : val_hysical_angle_loss,
            "val_dorothy_dialogue_move_loss" : val_dorothy_dialogue_move_loss,
            "val_dorothy_dialogue_slot_type_loss" : val_dorothy_dialogue_slot_type_loss,
            
            "val_dorothy_dialogue_slot_content_loss" : val_dorothy_dialogue_slot_content_loss,
            "val_wizard_dialogue_move_loss":val_wizard_dialogue_move_loss,
            "val_wizard_dialogue_slot_type_loss":val_wizard_dialogue_slot_type_loss,
            "val_wizard_dialogue_slot_content_loss":val_wizard_dialogue_slot_content_loss,
        })

        torch.save(model.cpu().state_dict(),  os.path.join(args.out_path,run_pth,f'{epoch}.torch'))
        model = model.to(args.device, non_blocking=True)
        if prev_loss > np.mean(loss_lst):
            prev_loss = np.mean(loss_lst)
            epochs_since_improvement = 0
            print('^', file=out_file)
        else:
            epochs_since_improvement += 1
            print(file=out_file)

        if epoch > min_num_epochs and epochs_since_improvement > max_wait_time:
            break


        start = time.time()

    if close_file:
        out_file.close()

if __name__ == "__main__":


    start = time.time()
    parser = ArgumentParser(description='TOTO')

    parser.add_argument('--dataset_path', type=str, default='arc_data',
                    help='Path to dataset')
    parser.add_argument('--out_path', type=str, default='exps',
                    help='Path to output directory, default is exps')
    parser.add_argument('--lr', type=float, default=1e-6,
                    help='learning rate setup')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='learning rate weight decay')

    parser.add_argument('--map_path', type=str, default='data/towns',
                    help='Path to dataset')

    parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use')


    parser.add_argument('--belief_out', default=True)



    parser.add_argument('--seed', type=int, default=0,
                    help='Path to saved model')

    parser.add_argument('--ablation', type=int, default=0,
                    help='Which ablation to run sequentially [0..7 or None]')
    parser.add_argument('--buffer_size', type=int, default=100,
                    help='length of image buffer')
    parser.add_argument('--buffer_step', type=int, default=4,
                    help='step of image buffer')
    parser.add_argument('--type_loss_weight', type=float, default=1e-2,
                    help='dialogue slot type loss weight')
    parser.add_argument('--angle_loss_weight', type=float, default=1e-2,
                    help='physical move angle loss weight')
    parser.add_argument('--content_loss_weight', type=float, default=2e-3,
                    help='dialogue slot type loss weight')
    args = parser.parse_args()
    print("TOTO training")
    config={
        "lr": args.lr,
        "seed": args.seed,
        "ablation":args.ablation,
        "angle_loss_weight":args.angle_loss_weight,
        "type_loss_weight": args.type_loss_weight,
        "content_loss_weight":args.content_loss_weight
    }
    run = wandb.init("TOTO",config=config)
    print('Train Games',end=' ')
    # train_games = Pool(10).map(partial(Game, args=args),train_game_paths)\
    train_events=[]
    train_path=os.path.join(args.dataset_path,"train")
    for path in tqdm(os.listdir(train_path)):
        train_events+=(Game(os.path.join(train_path,path),args=args).output_events)
    print('Loaded')

    print('Val Games',end=' ')
    val_events=[]
    val_path=os.path.join(args.dataset_path,"val")
    for path in tqdm(os.listdir(val_path)):
        val_events+=(Game(os.path.join(val_path,path),args=args).output_events)

    format_time = lambda x: f'{int(x)//3600:02d}:{int(x)%3600//60:02d}:{int(x%60):02d}'
    
    print('Loaded')
    print(format_time(time.time()-start))

    # exit()
    print("Start Training")
    if args.ablation is not None :
        main(args.ablation, train_events, val_events, args)