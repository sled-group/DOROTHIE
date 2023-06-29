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
import json
import random


import os, sys
import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob

from tqdm import tqdm

from nn.action_encoder import ActionEncoder
# from nn.knowledge_encoder import LaneEncoder
from ontology.agent import PhysicalAction, DialogueMove, DialogueSlot
from ontology.geometry import Lane

from data.dataset import Game

from model.toto import TOTO
import torch
from torch import optim, nn
import math
from multiprocessing import Pool, Manager

from functools import partial
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score,recall_score
from copy import deepcopy

from random import shuffle

frame_buffer_size = 100
frame_buffer_step = 4
event_batch_size = 256
format_time = lambda x: f'{int(x)//3600:02d}:{int(x)%3600//60:02d}:{int(x%60):02d}'



def do_epoch(args, model, optimizer, criterion_cel, criterion_mse,criterion_bce, events, do_eval=False):
    times = []
    loss_lst = []
    own_pos_err = []
    own_angle_err = []
    goal_pos_err = []
    act_type_pairs = []
    act_angle_err = []
    act_per_move_accs=[]
    for i in range(len(DialogueMove.__members__)):
        act_per_move_accs+=[]

    dorothy_dlg_move_pairs = []
    dorothy_type_rec = []
    dorothy_slot_rec = []
    wizzard_dlg_move_pairs = []
    wizzard_type_rec = []
    wizzard_slot_rec = []
    act_total_acc = []
    dorothy_total_acc=[]
    wizzard_total_acc =[]
    dorothy_slot_rec=[]
    wizzard_slot_rec=[]
    dorothy_total_slot_rec=[]
    wizzard_total_slot_rec=[]
    # num_epochs = 1

    for event in tqdm(events):
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
                loss+=1e-3*torch.sqrt(criterion_mse(belief ,gt_belief )) * int(args.belief_out)

                if gt_physical_type !=None:
                    # print(physical_type.shape,gt_physical_type)
                    loss+=criterion_cel(physical_type .float(), gt_physical_type ) * int(args.act_out) 

                if gt_physical_angle !=None:
                    loss+=1e-2*torch.sqrt(criterion_mse(physical_angle , gt_physical_angle .float().reshape(-1,1))) * int(args.act_out)
                
                if gt_dorothy_dialogue_move !=None:
                    loss+=criterion_cel(dorothy_dialogue_move .float(), gt_dorothy_dialogue_move ) * int(args.dorothy_out)
                
                if gt_dorothy_dialogue_type !=None:
                    loss+=1e-2*criterion_bce(dorothy_dialogue_type.float(), gt_dorothy_dialogue_type.float() ) * int(args.dorothy_out)
                    loss+=2e-3*criterion_bce(dorothy_dialogue_slot.float(), gt_dorothy_dialogue_slot.float() ) * int(args.dorothy_out)
                if gt_wizzard_dialogue_move !=None:
                    # print(gt_wizzard_dialogue_move)
                    # print(wizzard_dialogue_move.shape)
                    loss+=criterion_cel(wizzard_dialogue_move.float(), gt_wizzard_dialogue_move) * int(args.wizzard_out) 
                
                if gt_wizzard_dialogue_type !=None:
                    loss+=1e-2*criterion_bce(wizzard_dialogue_type .float(), gt_wizzard_dialogue_type.float()  ) * int(args.wizzard_out)
                    loss+=2e-3*criterion_bce(wizzard_dialogue_slot.float(), gt_wizzard_dialogue_slot.float() ) * int(args.dorothy_out)
                    
                # print(loss)
                if not do_eval:    
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                loss_lst.append(loss.item())


                slot_eval_fun = lambda x,y: [(precision_score(a,b,zero_division=1),recall_score(a,b,zero_division=1)) for a,b in zip(x,y)]
                belief_sqr_err = (belief-gt_belief)**2
                own_pos_err += list(torch.sqrt(belief_sqr_err[:,0]+belief_sqr_err[:,1]).cpu().data.numpy())
                own_angle_err += list(torch.sqrt(belief_sqr_err[:,2]%180).cpu().data.numpy())
                goal_pos_err += list(torch.sqrt(belief_sqr_err[:,3]+belief_sqr_err[:,4]).cpu().data.numpy())
        

                if gt_physical_type !=None:
                    act_type_pairs += list(zip(torch.argmax(physical_type,axis=-1).cpu().data.numpy(),gt_physical_type.cpu().data.numpy()))
                   
                    act_angle_err +=list((torch.abs(physical_angle-gt_physical_angle)<15).cpu().data.numpy())
                    if torch.eq(torch.argmax(physical_type,axis=-1),gt_physical_type):

                        if (torch.abs(physical_angle-gt_physical_angle)<15)[0]:
                            act_total_acc += [1]
                        else:
                            act_total_acc += [0]

                    else:
                        
                        act_total_acc += [0]
                
                if gt_dorothy_dialogue_move !=None:
                    dorothy_dlg_move_pairs += list(zip(torch.argmax(dorothy_dialogue_move,axis=-1).cpu().data.numpy(),gt_dorothy_dialogue_move.cpu().data.numpy()))
                    # print(dorothy_dialogue_type.shape,torch.argmax(dorothy_dialogue_type, axis=-1),gt_dorothy_dialogue_type)
                    dorothy_type_rec += slot_eval_fun((dorothy_dialogue_type>0).cpu().data.numpy()*1,gt_dorothy_dialogue_type.cpu().data.numpy())
                    if gt_dorothy_dialogue_type[0,0]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[0:10]>0).cpu().data.numpy()*1, gt_dorothy_dialogue_slot[0:10].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,1]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[10:26]>0).cpu().data.numpy()*1, gt_dorothy_dialogue_slot[10:26].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,2]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[26:32]>0).cpu().data.numpy()*1, gt_dorothy_dialogue_slot[26:32].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,3]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[32:44]>0).cpu().data.numpy()*1, gt_dorothy_dialogue_slot[32:44].cpu().data.numpy() )
                    if gt_dorothy_dialogue_type[0,4]==1:
                        dorothy_slot_rec+=slot_eval_fun((dorothy_dialogue_slot[44:]>0).cpu().data.numpy()*1, gt_dorothy_dialogue_slot[44:].cpu().data.numpy() )
                    dorothy_total_slot_rec+=[torch.all(torch.eq((dorothy_dialogue_slot>0)*1,gt_dorothy_dialogue_slot)).cpu().numpy()]
                   # print(torch.eq((dorothy_dialogue_type>0)*1,gt_dorothy_dialogue_type))
                    # print(torch.eq((dorothy_dialogue_slot>0)*1,gt_dorothy_dialogue_slot))
                    # print(torch.argmax(dorothy_dialogue_move,axis=-1),gt_dorothy_dialogue_move)
                    # print(torch.eq((dorothy_dialogue_slot>0)*1,gt_dorothy_dialogue_slot))
                    if torch.eq(torch.argmax(dorothy_dialogue_move,axis=-1),gt_dorothy_dialogue_move):
                        if gt_dorothy_dialogue_move==0 or gt_dorothy_dialogue_move==3 or gt_dorothy_dialogue_move==7 or gt_dorothy_dialogue_move==9 or gt_dorothy_dialogue_move==11 or gt_dorothy_dialogue_move==12:
                            dorothy_total_acc+=[1]
                        else:
                            if torch.all(torch.eq((dorothy_dialogue_type>0)*1,gt_dorothy_dialogue_type)):
                                is_right=True
                                if (gt_dorothy_dialogue_type[0,0]==1) and not (torch.all(torch.eq((dorothy_dialogue_slot[0:10]>0)*1,gt_dorothy_dialogue_slot[0:10]))):
                                    is_right=False
                                if (gt_dorothy_dialogue_type[0,1]==1) and not (torch.all(torch.eq((dorothy_dialogue_slot[10:26]>0)*1,gt_dorothy_dialogue_slot[10:26]))):
                                    is_right=False
                                if (gt_dorothy_dialogue_type[0,2]==1) and not (torch.all(torch.eq((dorothy_dialogue_slot[26:32]>0)*1,gt_dorothy_dialogue_slot[26:32]))):
                                    is_right=False
                                if (gt_dorothy_dialogue_type[0,3]==1) and not (torch.all(torch.eq((dorothy_dialogue_slot[32:44]>0)*1,gt_dorothy_dialogue_slot[32:44]))):
                                     is_right=False
                                if (gt_dorothy_dialogue_type[0,4]==1) and not (torch.all(torch.eq((dorothy_dialogue_slot[44:]>0)*1,gt_dorothy_dialogue_slot[44:]))):
                                    is_right=False
                                if is_right:
                                    dorothy_total_acc+=[1]
                                else:
                                    dorothy_total_acc+=[0]
                            else:
                                dorothy_total_acc+=[0]
                    else:
                        dorothy_total_acc+=[0]
                    # if torch.argmax(physical_type,axis=-1).cpu().data.numpy()[0,]==gt_physical_type.cpu().data.numpy()[0]:
                    #     if (torch.abs(physical_angle-gt_physical_angle)<15).cpu().data.numpy()[0]:
                    #         act_angle_err += [1]
                    #     else:
                    #         act_angle_err += [0]

                    # else:
                        
                    #     act_angle_err += [0]
                if gt_wizzard_dialogue_move !=None:
                    wizzard_dlg_move_pairs += list(zip(torch.argmax(wizzard_dialogue_move,axis=-1).cpu().data.numpy(),gt_wizzard_dialogue_move.cpu().data.numpy()))
                if gt_wizzard_dialogue_type !=None:
                    wizzard_type_rec += slot_eval_fun((wizzard_dialogue_type>0).cpu().data.numpy()*1,gt_wizzard_dialogue_type.cpu().data.numpy())
                    if gt_wizzard_dialogue_type[0,0]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[0:10]>0).cpu().data.numpy()*1, gt_wizzard_dialogue_slot[0:10].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,1]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[10:26]>0).cpu().data.numpy()*1, gt_wizzard_dialogue_slot[10:26].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,2]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[26:32]>0).cpu().data.numpy()*1, gt_wizzard_dialogue_slot[26:32].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,3]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[32:44]>0).cpu().data.numpy()*1, gt_wizzard_dialogue_slot[32:44].cpu().data.numpy() )
                    if gt_wizzard_dialogue_type[0,4]==1:
                        wizzard_slot_rec+=slot_eval_fun((wizzard_dialogue_slot[44:]>0).cpu().data.numpy()*1, gt_wizzard_dialogue_slot[44:].cpu().data.numpy() )
                    wizzard_total_slot_rec+=[torch.all(torch.eq((wizzard_dialogue_slot>0)*1,gt_wizzard_dialogue_slot)).cpu().numpy()]
                    if torch.eq(torch.argmax(wizzard_dialogue_move,axis=-1),gt_wizzard_dialogue_move):
                        if gt_wizzard_dialogue_move==0 or gt_wizzard_dialogue_move==3 or gt_wizzard_dialogue_move==7 or gt_wizzard_dialogue_move==9 or gt_wizzard_dialogue_move==11 or gt_wizzard_dialogue_move==12:
                            wizzard_total_acc+=[1]
                        else:
                            if torch.all(torch.eq((wizzard_dialogue_type>0)*1,gt_wizzard_dialogue_type)):
                                is_right=True
                                if (gt_wizzard_dialogue_type[0,0]==1) and not (torch.all(torch.eq((wizzard_dialogue_slot[0:10]>0)*1,gt_wizzard_dialogue_slot[0:10]))):
                                    is_right=False
                                if (gt_wizzard_dialogue_type[0,1]==1) and not (torch.all(torch.eq((wizzard_dialogue_slot[10:26]>0)*1,gt_wizzard_dialogue_slot[10:26]))):
                                    is_right=False
                                if (gt_wizzard_dialogue_type[0,2]==1) and not (torch.all(torch.eq((wizzard_dialogue_slot[26:32]>0)*1,gt_wizzard_dialogue_slot[26:32]))):
                                    is_right=False
                                if (gt_wizzard_dialogue_type[0,3]==1) and not (torch.all(torch.eq((wizzard_dialogue_slot[32:44]>0)*1,gt_wizzard_dialogue_slot[32:44]))):
                                     is_right=False
                                if (gt_wizzard_dialogue_type[0,4]==1) and not (torch.all(torch.eq((wizzard_dialogue_slot[44:]>0)*1,gt_wizzard_dialogue_slot[44:]))):
                                    is_right=False
                                if is_right:
                                    wizzard_total_acc+=[1]
                                else:
                                    wizzard_total_acc+=[0]
                            else:
                                wizzard_total_acc+=[0]
                    else:
                        wizzard_total_acc+=[0]
                   
        
        
        
    return loss_lst, own_pos_err, own_angle_err, goal_pos_err, act_type_pairs, act_angle_err, dorothy_dlg_move_pairs, dorothy_type_rec,dorothy_slot_rec, wizzard_dlg_move_pairs, wizzard_type_rec,wizzard_slot_rec, act_total_acc,dorothy_total_acc,wizzard_total_acc,dorothy_total_slot_rec,wizzard_total_slot_rec

def main(proc_idx, events, args):

    if proc_idx is None:
        out_file = sys.stdout
        close_file = False
    else:
        
        if args.seen:
            out_file = open(f'seen_exception.csv','a')
        else:
            out_file = open(f'unseen_exception.csv','a')
        close_file = True
        # args.save_path = f'/gpfs/accounts/chaijy_root/chaijy0/owenhji/model_path/ablation{proc_idx}_seed{args.seed}.torch'

        args.save_path = os.path.join(args.out_path,f'ablation{proc_idx}_seed{args.seed}.torch')
        # args.save_path = f'out{proc_idx}.torch'


        if 'cuda' in args.device and args.parallel:
            args.device = f'cuda:{proc_idx%2}'

        

        if proc_idx == 1:
            pass
        else:
            args.belief_out = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # print(args.dataset_path)
    # print(game_paths)

    model = TOTO(args).to(args.device, non_blocking=True)
    model.train()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.load_state_dict(torch.load(args.save_path))
    model.eval()

    # print(game_paths)
    # exit()
    # games = [Game(game_path,args) for game_path in game_paths]




    learning_rate = 1e-4
    num_epochs = 1000#2#1#
    weight_decay=1e-4

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

    max_num_epochs = 40#30
    min_num_epochs = 20#15
    start = time.time()
    prev_loss = 1e10
    epochs_since_improvement = 0
    max_wait_time = 5




    output = do_epoch(args, model, optimizer, criterion_cel, criterion_mse,criterion_bce, events, do_eval=True)
    loss_lst, own_pos_err, own_angle_err, goal_pos_err, act_type_pairs, act_angle_err, dorothy_dlg_move_pairs, dorothy_type_rec,dorothy_slot_rec, wizzard_dlg_move_pairs, wizzard_type_rec,wizzard_slot_rec, act_total_acc,dorothy_total_acc,wizzard_total_acc,dorothy_total_slot_rec,wizzard_total_slot_rec = output
        # for o in output:
        #     print(len(o))


    times.append(time.time()-start)
    # dorothy_total_slot_prec,dorothy_total_slot_rec=zip(*dorothy_total_slot_rec)
    # wizzard_total_slot_prec,wizzard_total_slot_rec=zip(*wizzard_total_slot_rec)

    print(  f'ablation{proc_idx}_seed{args.seed},',
            f'{np.mean(loss_lst):.3f},',
            f'{np.std(loss_lst):.3f},',
            f'{np.mean(own_pos_err):.3f},',
            f'{np.mean(own_angle_err):.3f},',
            f'{np.mean(goal_pos_err):.3f},',
            f'{accuracy_score(*zip(*act_type_pairs)):.3f},',
            f'{np.mean(act_angle_err):.3f},',
            f'{np.mean(act_total_acc):.3f},',
            f'{accuracy_score(*zip(*dorothy_dlg_move_pairs)):.3f},',
            f'{np.mean(dorothy_type_rec):.3f},',
            f'{np.mean(dorothy_slot_rec):.3f},',
            f'{np.mean(dorothy_total_slot_rec):.3f},',
            # f'{np.mean(dorothy_total_slot_prec):.3f},',
            f'{np.mean(dorothy_total_acc):.3f},',
            f'{accuracy_score(*zip(*wizzard_dlg_move_pairs)):.3f},',
            f'{np.mean(wizzard_type_rec):.3f},',
            f'{np.mean(wizzard_slot_rec):.3f},',
            f'{np.mean(wizzard_total_slot_rec):.3f},',
            # f'{np.mean(wizzard_total_slot_prec):.3f},',
            f'{np.mean(wizzard_total_acc):.3f},',
            file=out_file,
            flush=True
            )
    

    if close_file:
        out_file.close()

if __name__ == "__main__":


    start = time.time()
    parser = ArgumentParser(description='TOTO')

    parser.add_argument('--dataset_path', type=str, default='arc_data',
                    help='Path to dataset')
    parser.add_argument('--out_path', type=str, default='exps',
                    help='Path to output directory, default is exps')

    parser.add_argument('--map_path', type=str, default='data/towns',
                    help='Path to dataset')

    parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use')
    parser.add_argument('--save_path', type=str, default='out.torch',
                    help='Path to saved model')

    parser.add_argument('--video', default=True)
    parser.add_argument('--def_detr', default=True, help='Flag for using rbg embedding using Deformable DETR')
    parser.add_argument('--detr', default=True, help='Flag for using rbg embedding using off-the-shelf DETR')
    parser.add_argument('--dialogue', default=True)
    parser.add_argument('--dialogue_moves', default=True)
    parser.add_argument('--speech', default=True)
    parser.add_argument('--control', default=True)
    parser.add_argument('--interaction', default=True)
    parser.add_argument('--belief', default=True)
    parser.add_argument('--objects', default=True)


    parser.add_argument('--belief_out', default=True)
    parser.add_argument('--act_out', default=True)
    parser.add_argument('--dorothy_out', default=True)
    parser.add_argument('--wizzard_out', default=True)
    parser.add_argument('--seen', type=int, default=1)


    parser.add_argument('--parallel', default=False)


    parser.add_argument('--seed', type=int, default=0,
                    help='Path to saved model')

    parser.add_argument('--ablation', type=int, default=None,
                    help='Which ablation to run sequentially [0..7 or None]')

    args = parser.parse_args()

    print('Games',end=' ')

    print('Test Games',end=' ')
    if args.seen:
        test_game_paths=os.listdir(os.path.join(args.dataset_path,"test","seen"))
        test_game_paths=[os.path.join(args.dataset_path,"test","seen",path) for path in test_game_paths]
    else:
        test_game_paths=os.listdir(os.path.join(args.dataset_path,"test","unseen"))
        test_game_paths=[os.path.join(args.dataset_path,"test","unseen",path) for path in test_game_paths]
    test_events=[]
    for path in tqdm(test_game_paths):
        test_events+=(Game(path,args=args).exception_events)

    format_time = lambda x: f'{int(x)//3600:02d}:{int(x)%3600//60:02d}:{int(x%60):02d}'
    print('Loaded')
    print(format_time(time.time()-start))
    if args.seen:
        with open(f'seen_exception.csv','w') as f:

            print('ablation,loss,,belief,,,Act,,,dlg Dorothy,,,,,dlg Wizzard,,,,', file=f, flush=True)
            print('ablation,avg,stdev,own err,angle,gaol err,type,angle,acc,move,type,slot,total_prec,total_rec,acc,move,type,slot,total_prec,total_rec,acc', file=f, flush=True)
  
    else:
        with open(f'unseen_exception.csv','w') as f:

            print('ablation,loss,,belief,,,Act,,,dlg Dorothy,,,,,dlg Wizzard,,,,', file=f, flush=True)
            print('ablation,avg,stdev,own err,angle,gaol err,type,angle,acc,move,type,slot,total_prec,total_rec,acc,move,type,slot,total_prec,total_rec,acc', file=f, flush=True)
 
    # for i in range(13,14):
    #     for j in range(0):
    args.ablation=0
    args.seed=0
    main(args.ablation, test_events, args)
    # else:
    #     torch.multiprocessing.spawn(main, args=(val_games, args), nprocs=1, join=True)
    