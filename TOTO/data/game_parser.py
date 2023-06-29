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


import gc

import os
import json
import pprint
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import soundfile as sf
import pickle as pkl
import _pickle as cpkl
import torch
import cv2
import math
from multiprocessing import Manager

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from queue import Queue

import networkx as nx

from ontology.carla import SemanticClass
from ontology.agent import DialogueMoveCodingScheme, PhysicalAction, DialogueSlot, PA_ARGS,DialogueMove, Landmarks,StreetNames,GoalStatus
from ontology.geometry import Lane, Landmark

from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2Processor, HubertModel
from transformers import DetrModel, DetrFeatureExtractor

import torchaudio.functional as taf


def load_image(file_path):
    # print(file_path, end=' ')
    img = Image.open(file_path)
    if not (480 in img.size and 270 in img.size) :
        old_size = img.size
        img = img.resize((1920//4,1080//4))
        img.save(file_path)
        img = np.asarray(img)
        print(file_path, old_size, '->', img.shape)
    # exit()
    return np.asarray(Image.open(file_path))


def dict_to_list(dct, dim, def_val=None):
    retval = [def_val]*dim
    for idx in range(dim):
        if idx in dct:
            retval[idx] = dct[idx]
    return retval

def get_road(G, road_id):
    node_list = []
    for node in G.nodes:
        if G.nodes[node]["opendriveid"]["road_id"] == road_id:
            node_list.append(node)
    return node_list


def get_junction(G, junction_id):
    node_list = []
    for node in G.nodes:
        if G.nodes[node]["opendriveid"]["junction_id"] == junction_id:
            node_list.append(node)
    return node_list


class Game(object):

    def __init__(self, session_id, args):
        print(session_id)
        # if os.path.exists(os.path.join(args.dataset_path,session_id,"event_emb.pkl")):
        #     os.remove(os.path.join(args.dataset_path,session_id,"event_emb.pkl"))
        # return 
        # if True:
        if not os.path.exists(os.path.join(args.dataset_path,session_id,"event_emb.pkl")):
            self.session_id = session_id
            self.args = args
            self.dataset_path = args.dataset_path
            self.map_path = args.map_path

            self.config = None
            self.map = None
            self.semaphor = False
            # self.log = Queue()
            # self.trajectory = Queue()

            self.parse_config()
            self.parse_log()

            self.parse_config()
            self.parse_map()
            self.parse_trajectory()
            self.load_camera()
            self.load_dialogue()
            self.load_speech()
            self.__length = max(
                max(self.trajectory.keys()),
                max(self.rgb.keys()),
                # max(self.depth.keys()),
                # max(self.segmentation.keys()),
                max(self.log.keys()),
                # max(self.rgb_detr.keys()),
                max(self.dialogue.keys()),
                max(self.speech.keys()),
                )

            self.trajectory = dict_to_list(self.trajectory, self.__length, {'frame': 0, 'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'junction_id': -1, 'road_id': -1, 'lane_id': -1})
            self.rgb = dict_to_list(self.rgb, self.__length, np.zeros(list(self.rgb.values())[0].shape))
            # self.depth = dict_to_list(self.depth, self.__length, np.zeros((224,224)))
            # self.segmentation = dict_to_list(self.segmentation, self.__length, np.zeros((224,224,3)))
            self.log = dict_to_list(self.log, self.__length, [])
            # self.dialogue = dict_to_list(self.dialogue, self.__length, None)
            # self.rgb_detr = dict_to_list(self.rgb_detr, self.__length, 0*list(self.rgb_detr.values())[0])
            self.rgb_def_detr = dict_to_list(self.rgb_def_detr, self.__length, 0*list(self.rgb_def_detr.values())[0])
            self.speech = dict_to_list(self.speech, self.__length, np.zeros(list(self.speech.values())[0].shape))
            self.rgb=np.array(self.rgb)
            self.speech=np.array(self.speech)
            self.dialogue_embeddings = [
                (idx,torch.cat([torch.tensor(val)])) for idx, val in self.dialogue.items()
            ]
            # print(self.output_events)
            print("begin_parse")
            self.parse_traj()
            self.parse_event()
            print("end_parse")
            with open(os.path.join(args.dataset_path,session_id,"event_emb.pkl"),"wb") as f:
                pkl.dump(self.output_events,f)
            # exit()
        self.output_events=pkl.load(open(os.path.join(args.dataset_path,session_id,"event_emb.pkl"),'rb'))

        self.__index = 0
    def parse_traj(self):
        for traj in self.trajectory:
            if traj["road_id"] not in self.all_road_id:
                traj["road_id"]=0
            else:
                traj["road_id"]=self.all_road_id[traj["road_id"]]
            if traj["junction_id"] not in self.all_junction_id:
                traj["junction_id"]=0
            else:
                traj["junction_id"]=self.all_junction_id[traj["junction_id"]]

    def parse_event(self):
        self.count=0
        
        for event in self.output_events:
            event["lanes"]=self.lanes
            event["edge_index"]=self.edge_index
            event["landmarks"]=self.known_landmarks
            event["landmark_name"]=self.landmark_names
            event["streets"]=self.street_ids
            event["street_name"]=self.street_name_emb
            event["obj_def_detr"]=torch.tensor(self.rgb_def_detr[event["frame"]])
            if event["frame"]<self.args.buffer_size:
                idxs=list(range(event["frame"],0,-self.args.buffer_step))
                idxs.reverse()
                event["rgb"]=torch.tensor(self.rgb[idxs])
                event["belief"]=[self.trajectory[i] for i in idxs]
                event["rgb_idx"]=idxs
            else:
                idxs=list(range(event["frame"],event["frame"]-self.args.buffer_size,-self.args.buffer_step))
                idxs.reverse()
                event["rgb"]=torch.tensor(self.rgb[idxs])
                event["belief"]=[self.trajectory[i] for i in idxs]
                event["rgb_idx"]=idxs
            event["speech"]=torch.tensor(self.speech[event["frame"]])
            dialogue_idxs, dialogue = zip(*self.dialogue_embeddings)
            if event['type'] == 'DialogueMove' and event['to'] == 'dorothy':
                dlg_idx = [i for i,di in enumerate(dialogue_idxs) if di < event["frame"]]
                event["dialog_history_idx"]=[di for i,di in enumerate(dialogue_idxs) if di < event["frame"]]
            else:
                dlg_idx = [i for i,di in enumerate(dialogue_idxs) if di <= event["frame"]]
                event["dialog_history_idx"]=[di for i,di in enumerate(dialogue_idxs) if di <= event["frame"]]
            # print(dlg_idx)
            event["dialog_history"]=torch.stack([dialogue[i]for i in dlg_idx],axis=0).detach() if dlg_idx else torch.zeros([0])
            event["dialog_history_dir"]=[]
            event["dialog_history_move"]=[]
            event["dialog_history_type"]=[]
            event["dialog_history_slot"]=[]
            event["dialog_move_idx"]=[]

            for event_pst in self.output_events:
                if event_pst is not None and event_pst["frame"] in event["dialog_history_idx"] and  event_pst['type'].strip() == 'DialogueMove':
                    event["dialog_history_dir"].append([
                        int(event_pst['from'] == 'dorothy'),
                        int(event_pst['to']   == 'dorothy'),
                    ])
                    if "StreetName" in  event_pst['act']['slots']:
                        event_pst['act']['slots']["Street"]=event_pst['act']['slots']["StreetName"]
                        del event_pst['act']['slots']["StreetName"]
                    event["dialog_move_idx"].append(event_pst["frame"])
                    event["dialog_history_move"].append(int(eval(f"DialogueMove.{event_pst['act']['move']}")))
                    types = [0]*(len(DialogueSlot.__members__)-1)
                    actions=[0]*(len(PhysicalAction.__members__))
                    streets=[0]*(len(StreetNames.__members__))
                    statuses=[0]*(len(GoalStatus.__members__))
                    landmarks=[0]*(len(Landmarks.__members__))
                    objects=[0]*(len(SemanticClass.__members__))
                    for x,type_slot in event_pst['act']['slots'].items():
                        types[int(eval(f"DialogueSlot.{x}"))-1] = 1
                        for one_slot in type_slot:
                            if one_slot=="Vehicle":
                                one_slot="Vehicles"
                            if one_slot=="Completed":
                                one_slot="Complete"
                            if one_slot=="Seven Eleven":
                                one_slot="SevenEleven"
                            if one_slot=="Jturn":
                                one_slot="JTurn"
                            if x=="Action":
                                actions[int(eval(f"PhysicalAction.{one_slot}"))] = 1
                            if x=="Street":
                                streets[int(eval(f"StreetNames.{one_slot}"))] = 1
                            if x=="Status":
                                statuses[int(eval(f"GoalStatus.{one_slot}"))] = 1
                            if x=="Landmark":
                                landmarks[int(eval(f"Landmarks.{one_slot}"))] = 1
                            if x=="Object":
                                objects[int(eval(f"SemanticClass.{one_slot}"))] = 1
                    event["dialog_history_type"].append(types)
                    event["dialog_history_slot"].append(actions+streets+statuses+landmarks+objects)
            event["dialog_history_dir"] = torch.tensor(np.array(event["dialog_history_dir"])).int()
            event["dialog_history_move"] = torch.tensor(np.array(event["dialog_history_move"])).int()
            event["dialog_history_type"] = torch.tensor(np.array(event["dialog_history_type"])).int()
            event["dialog_history_slot"] = torch.tensor(np.array(event["dialog_history_slot"])).int()
            # print(len(event["dialog_history_idx"]),event["dialog_history_dir"].shape[0],event["dialog_history_idx"][-1])
            # print(event["dialog_history_idx"])
            # print(event["dialog_move_idx"])
            # assert len(event["dialog_history_idx"])==event["dialog_history_dir"].shape[0]
            event["act_history_idx"]=[]
            event["act_history_type"]=[]
            event["act_history_val"]=[]
            for event_pst in self.output_events:
                if event_pst is not None and event_pst["frame"]<event["frame"] and event_pst['type'].strip() == 'PhysicalAction':
                        # print(event)
                    event["act_history_idx"].append(event_pst["frame"])
                    event["act_history_type"].append(
                        int(eval(f"PhysicalAction.{event_pst['val']['act']}"))
                    )
                    # print(event["act_history_val"],event["frame"])
                    if event_pst['val']['act'] in ['Start', 'Stop', 'Unknown', 'LaneFollow']:
                        event["act_history_val"].append(0)
                    elif event_pst['val']['act']=="UTurn":
                        event["act_history_val"].append(-180.0)
                    elif event_pst['val']['act']=="LaneSwitch":
                        event["act_history_val"].append((event_pst['val']['slot_val']-2.5)*180)
                    else:
                        event["act_history_val"].append(float(event_pst['val']['slot_val']))
            event["act_history_type"]=torch.tensor(np.array(event["act_history_type"])).int()
            event["act_history_val"]=torch.tensor(np.array(event["act_history_val"])).int()
            
            game_trajectory = self.trajectory
            idx=event["frame"]
            num_frames=self.__length
            goal = [x for x in self.goals if x['start_frame'] <= idx and x['end_frame'] >= idx][-1:]

            norm_angle = lambda x: norm_angle(x-360) if x > 180 else (norm_angle(x+360) if x <= -180 else x)
            
            goal_x = game_trajectory[idx+1]['x']
            goal_y = game_trajectory[idx+1]['y']
                
            if goal:
                for x in self.config['environment']['landmarks']:
                    if x['name'] == goal[0]['landmark']:
                        goal_x = x['wp']['Transform']['Location']['x']
                        goal_y = x['wp']['Transform']['Location']['y']
                        break
            event["gt_belief"]=[
                game_trajectory[min(idx+1,num_frames)]['x']-game_trajectory[idx]['x'],
                game_trajectory[min(idx+1,num_frames)]['y']-game_trajectory[idx]['y'],
                norm_angle(game_trajectory[min(idx+1,num_frames)]['yaw']-game_trajectory[0]['yaw']),
                game_trajectory[idx]['x']-goal_x, 
                game_trajectory[idx]['y']-goal_y, 
            ]
            event["gt_belief"]=torch.tensor(event["gt_belief"])
            # if  event['type'].strip() == 'PhysicalAction' and 
            event["gt_physical_type"]= int(eval(f"PhysicalAction.{event['val']['act']}")) if event['type'].strip() == 'PhysicalAction' else None
            if event["gt_physical_type"]!=None:
                event["gt_physical_type"]=torch.tensor(event["gt_physical_type"])

            event["gt_physical_angle"] =(
                None if event["gt_physical_type"] is None else
                0 if not event['val']['act'] in ['LaneSwitch', 'JTurn'] else 
                (event['val']['slot_val']-2.5)*180 if not event['val']['act'] in ['JTurn'] else 
                norm_angle(event['val']['slot_val'])
            )
            if event["gt_physical_angle"] !=None:
                event["gt_physical_angle"] =torch.tensor(event["gt_physical_angle"] )
                
            if event['type'] == 'DialogueMove':
                if "StreetName" in  event['act']['slots']:
                        event['act']['slots']["Street"]=event['act']['slots']["StreetName"]
                        del event['act']['slots']["StreetName"]
                if event['from'] == 'dorothy':
                    types = [0]*(len(DialogueSlot.__members__)-1)
                    actions=[0]*(len(PhysicalAction.__members__))
                    streets=[0]*(len(StreetNames.__members__))
                    statuses=[0]*(len(GoalStatus.__members__))
                    landmarks=[0]*(len(Landmarks.__members__))
                    objects=[0]*(len(SemanticClass.__members__))
                    for x,type_slot in event['act']['slots'].items():
                        # print(x)
                        types[int(eval(f"DialogueSlot.{x}"))-1] = 1
                        for one_slot in type_slot:
                            if one_slot=="Vehicle":
                                one_slot="Vehicles"
                            if one_slot=="Completed":
                                one_slot="Complete"
                            if one_slot=="Seven Eleven":
                                one_slot="SevenEleven"
                            if one_slot=="Jturn":
                                one_slot="JTurn"
                            if x=="Action":
                                actions[int(eval(f"PhysicalAction.{one_slot}"))] = 1
                            if x=="Street":
                                streets[int(eval(f"StreetNames.{one_slot}"))] = 1
                            if x=="Status":
                                statuses[int(eval(f"GoalStatus.{one_slot}"))] = 1
                            if x=="Landmark":
                                landmarks[int(eval(f"Landmarks.{one_slot}"))] = 1
                            if x=="Object":
                                objects[int(eval(f"SemanticClass.{one_slot}"))] = 1
                    
                    event['gt_dorothy_dialogue_move'] =(int(eval(f"DialogueMove.{event['act']['move']}")))
                    event['gt_dorothy_dialogue_type'] =types
                    event['gt_dorothy_dialogue_slot'] =actions+streets+statuses+landmarks+objects
                    event['gt_wizzard_dialogue_move'] =None
                    event['gt_wizzard_dialogue_type'] =None
                    event['gt_wizzard_dialogue_slot'] =None
                else:
                    event['gt_dorothy_dialogue_move'] =None
                    event['gt_dorothy_dialogue_type'] =None
                    event['gt_dorothy_dialogue_slot'] =None

                    types = [0]*(len(DialogueSlot.__members__)-1)
                    actions=[0]*(len(PhysicalAction.__members__))
                    streets=[0]*(len(StreetNames.__members__))
                    statuses=[0]*(len(GoalStatus.__members__))
                    landmarks=[0]*(len(Landmarks.__members__))
                    objects=[0]*(len(SemanticClass.__members__))
                    for x,type_slot in event['act']['slots'].items():
                        types[int(eval(f"DialogueSlot.{x}"))-1] = 1
                        for one_slot in type_slot:
                            if one_slot=="Vehicle":
                                one_slot="Vehicles"
                            if one_slot=="Completed":
                                one_slot="Complete"
                            if one_slot=="Seven Eleven":
                                one_slot="SevenEleven"
                            if one_slot=="Jturn":
                                one_slot="JTurn"
                            if x=="Action":
                                actions[int(eval(f"PhysicalAction.{one_slot}"))] = 1
                            if x=="Street":
                                streets[int(eval(f"StreetNames.{one_slot}"))] = 1
                            if x=="Status":
                                statuses[int(eval(f"GoalStatus.{one_slot}"))] = 1
                            if x=="Landmark":
                                landmarks[int(eval(f"Landmarks.{one_slot}"))] = 1
                            if x=="Object":
                                objects[int(eval(f"SemanticClass.{one_slot}"))] = 1
                    
                    event['gt_wizzard_dialogue_move'] =(int(eval(f"DialogueMove.{event['act']['move']}")))
                    event['gt_wizzard_dialogue_type'] =types
                    event['gt_wizzard_dialogue_slot'] =actions+streets+statuses+landmarks+objects
            else:
                event['gt_dorothy_dialogue_move'] =None
                event['gt_dorothy_dialogue_type'] =None
                event['gt_dorothy_dialogue_slot'] =None
                event['gt_wizzard_dialogue_move'] =None
                event['gt_wizzard_dialogue_type'] =None
                event['gt_wizzard_dialogue_slot'] =None
            if event['gt_dorothy_dialogue_move'] !=None:
                event['gt_dorothy_dialogue_move'] =torch.tensor(event['gt_dorothy_dialogue_move'])
            if event['gt_dorothy_dialogue_type'] !=None:
                event['gt_dorothy_dialogue_type'] =torch.tensor(event['gt_dorothy_dialogue_type'])
            if event['gt_dorothy_dialogue_slot'] !=None:
                event['gt_dorothy_dialogue_slot'] =torch.tensor(event['gt_dorothy_dialogue_slot'] )
            if event['gt_wizzard_dialogue_move'] !=None:
                event['gt_wizzard_dialogue_move'] =torch.tensor(event['gt_wizzard_dialogue_move'])
            if event['gt_wizzard_dialogue_type'] !=None:
                event['gt_wizzard_dialogue_type'] =torch.tensor(event['gt_wizzard_dialogue_type'])
            if event['gt_wizzard_dialogue_slot'] !=None:
                event['gt_wizzard_dialogue_slot'] =torch.tensor(event['gt_wizzard_dialogue_slot'])




    def parse_config(self):
        with open(os.path.join(self.dataset_path, self.session_id, 'config.json')) as f:
            self.config = json.load(f)
        with open(os.path.join(self.dataset_path, self.session_id, 'plan.json')) as f:
            self.plan = json.load(f)
            self.goals = []
            node = None
            for g in self.plan:
                if g['status'] == 'Ongoing':
                    node = {
                        'start_frame': g['frame'],
                        'landmark': g['landmark']
                        }
                elif node is not None and node['landmark'] == g['landmark']:
                    node['end_frame'] = g['frame']
                    self.goals.append(node)
                    node = None
                else:
                    node = None

    def parse_map(self):
        with open(os.path.join(self.map_path, self.config['environment']['map']+'.json')) as f:
            self.map_data = json.load(f)
            self.map = nx.readwrite.adjacency_graph(self.map_data['map'], directed=True)
            self.meta_map = nx.readwrite.adjacency_graph(self.map_data['meta_map'], directed=True)
        with open(os.path.join('./data/asset_metadata.json')) as f:
            self.meta_assets = json.load(f)
            
        self.num_streets = self.map_data['num_streets']
        self.streets = self.map_data['streets']

        self.all_road_id = {-1:0}
        self.all_junction_id = {-1: 0}
        for node in self.map.nodes._nodes.values():
            r_id, j_id = node['opendriveid']['road_id'], node['opendriveid']['junction_id']
            if r_id not in self.all_road_id:
                self.all_road_id[r_id] = len(self.all_road_id)
            if j_id not in self.all_junction_id:
                self.all_junction_id[j_id] = len(self.all_junction_id)
        self.street_names=self.config["environment"]["street_names"]
        pkl_file = os.path.join(self.dataset_path,self.session_id,"street_name_emb.pkl")
        if not os.path.exists(pkl_file):
            self.street_name_emb = []
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()
            model.eval()

            for name in self.street_names:
                    encoded_dict = tokenizer.encode_plus(
                        name,  # Sentence to encode.
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        return_tensors='pt',  # Return pytorch tensors.
                    )
                    token_ids = encoded_dict['input_ids'].cuda()
                    segment_ids = torch.ones(token_ids.size()).long().cuda()
                    with torch.no_grad():
                        outputs = model(input_ids=token_ids, token_type_ids=segment_ids)
                    outputs = outputs[1][0].cpu().data.numpy()
                    self.street_name_emb.append(outputs)
            pkl.dump(self.street_name_emb,open(pkl_file,"wb"))
            print('Saved', pkl_file)
        self.street_name_emb=torch.tensor(np.array(pkl.load(open(pkl_file,'rb'))))[:self.num_streets,:]

        
        self.street_ids = []
        for i in range(self.num_streets):
            self.street_ids.append([])
            for r_id in self.streets['road_id'][i]:
                self.street_ids[-1].extend(get_road(self.map, r_id))
            for j_id in self.streets['junction_id'][i]:
                self.street_ids[-1].extend(get_junction(self.map, j_id))
            self.street_ids[-1] = list(set(self.street_ids[-1]))

        self.lanes = []
        for node in self.map.nodes._nodes.values():
            node['opendriveid']['road_id'] = self.all_road_id[node['opendriveid']['road_id']]
            node['opendriveid']['junction_id'] = self.all_junction_id[node['opendriveid']['junction_id']]
            self.lanes.append(Lane(node))
        self.edge_index = list(self.map.edges)



        self.landmarks = self.config['environment']['landmarks']
        self.known_landmarks = []
        for landmark in self.landmarks:
            if not landmark['is_hidden']:
                self.known_landmarks.append(Landmark({
                    'transform': {
                        'location': landmark['wp']['Transform']['Location'],
                        'rotation': landmark['wp']['Transform']['Rotation'],
                    },
                    'opendriveid': {#landmark['wp']['OpenDriveID'],#'{
                        'lane_id': landmark['wp']['OpenDriveID']['lane_id'],
                        'road_id': self.all_road_id[landmark['wp']['OpenDriveID']['road_id']],
                        'junction_id': self.all_junction_id[landmark['wp']['OpenDriveID']['junction_id']]
                    },
                    'asset': landmark['asset'],
                    'name': self.meta_assets[landmark['asset']]['name']
                }))

        pkl_file = os.path.join(self.dataset_path,self.session_id,"landmark_emb.pkl")
        # os.remove(pkl_file)
        if not os.path.exists(pkl_file):
            self.landmark_names = []
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()
            model.eval()

            for landmark in self.landmarks:
                if not landmark['is_hidden']:
                    encoded_dict = tokenizer.encode_plus(
                        self.meta_assets[landmark['asset']]['name'],  # Sentence to encode.
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        return_tensors='pt',  # Return pytorch tensors.
                    )
                    token_ids = encoded_dict['input_ids'].cuda()
                    segment_ids = torch.ones(token_ids.size()).long().cuda()
                    with torch.no_grad():
                        outputs = model(input_ids=token_ids, token_type_ids=segment_ids)
                    outputs = outputs[1][0].cpu().data.numpy()
                    self.landmark_names.append(outputs)
            pkl.dump(self.landmark_names,open(pkl_file,"wb"))
            print('Saved', pkl_file)
        self.landmark_names=torch.tensor(np.array(pkl.load(open(pkl_file,'rb'))))

    def parse_log(self):
        with open(os.path.join(self.dataset_path, self.session_id, 'annotated_log.json')) as f:
            self.log = {}
            self.output_events = []
            for entry in json.load(f):
                if not entry['frame'] in self.log:
                    self.log[entry['frame']] = []
                self.log[entry['frame']].append(entry)
                if entry['type'].strip() in ['DialogueMove', 'PhysicalAction'] and entry['flag']:
                    self.output_events.append(entry)

    def parse_trajectory(self):
        self.trajectory = {}
        with open(os.path.join(self.dataset_path, self.session_id, 'trajectory.csv')) as f:
            lines = list(f.readlines())
            header = lines[0].strip().split(', ')
            for i, line in enumerate(lines[1:]):
                if 'frame' in line: break
                data = [fun(x) for x,fun in zip(line.strip().split(', '),[int, float, float, float, int, int, int])]
                self.trajectory[data[0]] = (dict(zip(header, data)))

    def load_camera(self):
        files_fun = lambda s: sorted(glob(os.path.join(self.dataset_path, self.session_id, s)+'/*'))
        load_fun = lambda x: {int(y.split('/')[-1].split('.')[0]) : load_image(y) for y in files_fun(x)}

        pkl_file = os.path.join(self.dataset_path,self.session_id,"rgb_emb.pkl")
        # if os .path.exists(pkl_file):
        #     os.remove(pkl_file)
        if not os.path.isfile(pkl_file):
            # rgb = {k : cv2.resize(v, (224, 224)) for k, v in load_fun('rgb').items()}
            resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True).cuda()
            resnet50.eval()
            rgb = {}
            with torch.no_grad():
                for file in tqdm(files_fun('rgb')):
                    k = int(file.split('/')[-1].split('.')[0])
                    v = load_image(file)
                    rgb[k] = resnet50(torch.tensor(np.array([v])).permute(0,3,1,2)[:,0:3,:,:].float().cuda()).cpu().data.numpy()[0]
            pkl.dump(rgb,open(pkl_file,"wb"))
            print('Saved', pkl_file)
        self.rgb = pkl.load(open(pkl_file,'rb'))
        # print(self.rgb.keys())
        # for key in self.rgb.keys():
        #     if type(key)!='int':
        #         print(key)
        # self.depth = {k : cv2.resize(v, (224, 224)) for k, v in load_fun('depth').items()}
        # self.segmentation = {k : cv2.resize(v, (224, 224)) for k, v in load_fun('semantic_segmentation').items()}

        # pkl_file = os.path.join(self.dataset_path,self.session_id,"rgb_detr.pkl")
        # if not os.path.isfile(pkl_file):
        #     self.rgb_detr = {}
        #     feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        #     model = DetrModel.from_pretrained("facebook/detr-resnet-50").cuda()
        #     model.eval()
        #     # for key, image in self.rgb.items():
        #     for file in files_fun('rgb'):
        #         key = int(file.split('/')[-1].split('.')[0])
        #         image = cv2.resize(load_image(file), (224, 224))
        #         inputs = feature_extractor(images=image, return_tensors="pt")
        #         inputs = {k:v.cuda() for k,v in inputs.items()}
        #         with torch.no_grad():
        #             outputs = model(**inputs)
        #         last_hidden_states = outputs.last_hidden_state.cpu().data.numpy()
        #         self.rgb_detr[key] = last_hidden_states
        #     pkl.dump(self.rgb_detr,open(pkl_file,"wb"))
        #     print('Saved', pkl_file)
        # self.rgb_detr = pkl.load(open(pkl_file,'rb'))

        pkl_file = os.path.join(self.dataset_path,self.session_id,"rgb_def_detr.pkl")
        if not os.path.isfile(pkl_file):
            self.rgb_detr = {0: np.zeros((1,300,256))}
            feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
            model = DetrModel.from_pretrained("facebook/detr-resnet-50").cuda()
            model.eval()
            # for key, image in self.rgb.items():
            for file in files_fun('rgb'):
                key = int(file.split('/')[-1].split('.')[0])
                image = cv2.resize(load_image(file), (224, 224))
                inputs = feature_extractor(images=image, return_tensors="pt")
                inputs = {k:v.cuda() for k,v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state.cpu().data.numpy()
                self.rgb_detr[key] = last_hidden_states
            pkl.dump(self.rgb_detr,open(pkl_file,"wb"))
            print('Saved', pkl_file)
        self.rgb_def_detr = pkl.load(open(pkl_file,'rb'))
        for k in self.rgb_def_detr.keys():
            try:
                self.rgb_def_detr[k] = self.rgb_def_detr[k][-1:].cpu().data.numpy()
            except Exception as e:
                self.rgb_def_detr[k] = self.rgb_def_detr[k][-1:]

    def load_dialogue(self):
        pkl_file = os.path.join(self.dataset_path,self.session_id,"dialogue.pkl")
        # if True:
        if not os.path.isfile(pkl_file):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).cuda()
            model.eval()
            self.dialogue={}
            for frame in json.load(open(os.path.join(self.dataset_path, self.session_id, 'annotated_log.json'))):
                if frame['type'] == 'DialogueMove':
                    utterance = frame['utterance_gt'].replace('\u2019',',')
                    encoded_dict = tokenizer.encode_plus(
                        utterance,  # Sentence to encode.
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        return_tensors='pt',  # Return pytorch tensors.
                    )
                    token_ids = encoded_dict['input_ids'].cuda()
                    segment_ids = torch.ones(token_ids.size()).long().cuda()
                    with torch.no_grad():
                        outputs = model(input_ids=token_ids, token_type_ids=segment_ids)
                    outputs = outputs[1][0].cpu().data.numpy()
                    outputs = np.concatenate([[int(frame['from']=='dorothy')],[int(frame['to']=='dorothy')],outputs])
                    self.dialogue[frame['frame']] = outputs#{"from": frame['from'], "to": frame['to'], "val": outputs}
            pkl.dump(self.dialogue,open(pkl_file,"wb"))
            print('Saved', pkl_file)
        self.dialogue = pkl.load(open(pkl_file,'rb'))


    def load_speech(self):
        pkl_file = os.path.join(self.dataset_path,self.session_id,"speech.pkl")
        # if os.path.exists(pkl_file):
        #     os.remove(pkl_file)
        if not os.path.isfile(pkl_file):
            files_fun = lambda s: sorted(glob(os.path.join(self.dataset_path, self.session_id, s)+'/*'))
            load_fun = lambda x: {x.split('_')[-1].split('.')[0] : sf.read(x) for x in files_fun(x)}
            self.utterances = load_fun('utterance')
            processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
            model.eval()
            # print(self.log)
            with open(os.path.join(self.dataset_path, self.session_id, 'annotated_log.json')) as f:
                log = json.load(f)

            self.speech = {}
            for frame in tqdm(log):
                # print(frame_actions)
                # for frame in frame_actions:
                    # print(frame)
                    if frame['type'] == 'Speech':
                        utterance_idx = frame['utterance_filename'].split('_')[-1].split('.')[0]
                        utterance, sampling_rate = self.utterances[utterance_idx]
                        resample_rate = 16000
                        # print(utterance_idx)
                        utterance = taf.resample(
                            torch.tensor(utterance),
                            sampling_rate,
                            resample_rate,
                            lowpass_filter_width=128,
                            rolloff=0.99,
                            resampling_method="sinc_interpolation",
                        )
                        num_samples = utterance.shape[0]
                        window_proportion = (frame['frame']-frame['start_frame']+1)/(frame['end_frame']-frame['start_frame']+1)
                        window_size = int(num_samples*window_proportion)
                        heard_samples = utterance[:window_size].flatten() if len(utterance.shape)>1 else utterance[:window_size]
                        input_values = processor(heard_samples, sampling_rate=resample_rate, return_tensors="pt").input_values
                        # print(heard_samples.shape)
                        with torch.no_grad():
                            output = model(input_values).last_hidden_state.cpu().data.numpy()
                            self.speech[frame['frame']] = np.mean(output, axis=1)[0]
            pkl.dump(self.speech,open(pkl_file,"wb"))
            print('Saved', pkl_file)
        self.speech = pkl.load(open(pkl_file,'rb'))

    def __prep_trajectory_frame(self, idx):
        if idx in self.trajectory:
            return self.trajectory[idx] 
        else:
            return {'frame': idx, 'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'junction_id': -1, 'road_id': -1, 'lane_id': -1}

    def __prep_log_frame(self, idx):
        return self.log[idx]

    # def __iter__(self):
    #     self.__index = 0
    #     return self

    # def __next__(self):
    #     if self.__index < self.__length:
    #         retval = self[self.__index]
    #         self.__index += 1
    #         return retval
    #     self.__index = 0
    #     raise StopIteration

    def __len__(self):
        return self.__length

    def __getitem__(self, slc):
        while self.semaphor:
            pass
        self.semaphor = True

        sincos_fun = lambda i: torch.tensor([math.sin(2*torch.pi*i/self.__length), math.cos(2*torch.pi*i/self.__length)])
        get_frame = lambda lst, idx, default: torch.tensor(np.stack(lst[idx])) if isinstance(idx, slice) else torch.tensor(lst[idx])# if idx in lst else default)
        get_dialogue = lambda lst, idx: [get_dialogue_item(lst,i) for i in list(range(self.__length))[idx]] \
            if isinstance(idx, slice) else get_dialogue_item(lst,idx)
        get_dialogue_item = lambda lst, idx: torch.tensor(
            np.concatenate([sincos_fun(idx),lst[idx]])) \
            if not lst[idx] is None else \
            None#np.concatenate([[0,0],np.zeros(list(lst.values())[0]['val'].shape)])
        idx = slc
        retval = {
            'frame'        : idx,
            'rgb'          : get_frame(self.rgb, idx, np.zeros(self.rgb[0].shape, np.uint8)),
            'obj_def_detr'      : get_frame(self.rgb_def_detr, idx, np.zeros(self.rgb_def_detr[0].shape, np.uint8)),
            # 'depth'        : get_frame(self.depth, idx, np.zeros((224,224), np.uint8)),
            # 'segmentation' : get_frame(self.segmentation, idx, np.zeros((224,224,3), np.uint8)),
            # 'dialogue'     : get_dialogue(self.dialogue, idx),
            'speech'       : get_frame(self.speech, idx, 0),#np.zeros(list(self.speech.values())[0].shape)),
            # 'obj_detr'     : get_frame(self.rgb_detr, idx, np.zeros(self.rgb_detr[0].shape, np.uint8)),#*list(self.rgb_detr.values())[0]),
            'trajectory'   : self.__prep_trajectory_frame(idx),
            'log'          : self.__prep_log_frame(idx),
        }

        self.semaphor = False
        return retval

    def __setitem__(self, idx, val):
        pass

    def __delitem__(self, idx):
        pass


def Config(parser):
    parser.add_argument('--seed', default=0, dest='seed', help='random seed', type=int)


if __name__ == '__main__':

    dm = DialogueMoveCodingScheme()
    a, b = dm.decision_tree([1, 2, 3, 1, 1, 1, 1, 1, 1])

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    Config(parser)
    parser.add_argument('--dataset_path', default='../data/example/', dest='dataset_path', help='dataset folder')
    parser.add_argument('--map_path', default='../data/towns/', dest='map_path', help='map folder')
    args = parser.parse_args()

    pprint.pprint(args)

    games = [f for f in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, f))]
    print(games)
    session_id = games[0]
    dataset = Game(session_id, args)

    print()
