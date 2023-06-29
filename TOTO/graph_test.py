from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import json

import networkx as nx
import torch
import torch.nn.functional as F

from ontology.geometry import Lane, Landmark
from nn.graph_encoder import MapEncoder
from nn.knowledge_encoder import RoadEncoder, JunctionEncoder, OpenDriveEncoder, LandmarkEncoder


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


# Map Loading
# arc_data/1651242689


# with open(os.path.join('./data/example', '1652310899', 'config.json')) as f:
with open(os.path.join('./arc_data', '1651242689', 'config.json')) as f:
    config = json.load(f)
with open(os.path.join('./data/asset_metadata.json')) as f:
    meta_assets = json.load(f)
with open(os.path.join('./data/towns', config['environment']['map'] + '.json')) as f:
    map_data = json.load(f)

print(config['environment']['map'])

map = nx.readwrite.adjacency_graph(map_data['map'], directed=True)
num_streets = map_data['num_streets']
streets = map_data['streets']

street_names = config['environment']['street_names']
landmark_names = []

# Map Pre-processing

# TODO: Choose appropriate parameters
max_num_nodes = 486
max_num_roads = 259
max_num_junctions = 33
max_num_streets = 14
max_num_landmarks = 10


all_road_id = {-1: 0}
all_junction_id = {-1: 0}
for node in map.nodes._nodes.values():
    if node is None:
        continue
    r_id, j_id = node['opendriveid']['road_id'], node['opendriveid']['junction_id']
    if r_id not in all_road_id:
        all_road_id[r_id] = len(all_road_id)
    if j_id not in all_junction_id:
        all_junction_id[j_id] = len(all_junction_id)


lanes = []
for node in map.nodes._nodes.values():
    node['opendriveid']['road_id'] = all_road_id[node['opendriveid']['road_id']]
    node['opendriveid']['junction_id'] = all_junction_id[node['opendriveid']['junction_id']]
    lanes.append(Lane(node))
while len(lanes) < max_num_nodes:
    lanes.append(Lane(None))

edges = list(map.edges)
for i in range(max_num_nodes):
    edges.append((i, i))
edge_index = torch.tensor(edges).t()


street_ids = []
for i in range(num_streets):
    street_ids.append([])
    for r_id in streets['road_id'][i]:
        street_ids[-1].extend(get_road(map, r_id))
    for j_id in streets['junction_id'][i]:
        street_ids[-1].extend(get_junction(map, j_id))
    street_ids[-1] = list(set(street_ids[-1]))


# Map Encoding and Landmark Encoding

# TODO: Choose appropriate parameters
lane_embedding_dim = 64
landmark_embedding_dim = 57
id_embedding_dim = lane_embedding_dim - len(Lane()) + 2  # 50

road_encoder = RoadEncoder(max_num_nodes, id_embedding_dim)
junction_encoder = JunctionEncoder(max_num_junctions, id_embedding_dim)
id_encoder = OpenDriveEncoder(id_embedding_dim,
                              max_num_nodes,
                              max_num_junctions,
                              road_encoder, junction_encoder)
map_encoder = MapEncoder(lane_embedding_dim, id_encoder)
landmark_encoder = LandmarkEncoder(id_embedding_dim, id_encoder)

# print(all_road_id)
# print(all_junction_id)

# Map Feedforward

print(len(edge_index))
print(edge_index)
map_emb = map_encoder.forward(lanes, edge_index)  # [num_nodes (lanes), lane_embedding_dim]
street_emb = torch.zeros((num_streets, lane_embedding_dim))  # [num_streets, lane_embedding_dim]
for i, street_id in enumerate(street_ids):
    street_emb[i] = map_emb[street_id].mean(dim=0)

print(map_emb.shape)

print(street_emb.shape)

# TODO: Concat `street_emb` with `street_names`, dim=1
#       street_emb: [num_street, lane_embedding_dim]
#       street_names: str -> Text Encoder (Pre-trained): [num_streets, text_embedding_dim]
#       pad num_streets to max_num_streets (optional?)

# Landmark Feedforward

landmarks = config['environment']['landmarks']
known_landmarks = []
for landmark in landmarks:
    if not landmark['is_hidden']:
        # print(landmark)
        known_landmarks.append(Landmark({
            'transform': {
                'location': landmark['wp']['Transform']['Location'],
                'rotation': landmark['wp']['Transform']['Rotation'],
            },
            'opendriveid': {#landmark['wp']['OpenDriveID'],#'{
                'lane_id': landmark['wp']['OpenDriveID']['lane_id'],
                'road_id': all_road_id[landmark['wp']['OpenDriveID']['road_id']],
                'junction_id': all_junction_id[landmark['wp']['OpenDriveID']['junction_id']]
            },
            'asset': landmark['asset'],
            'name': meta_assets[landmark['asset']]['name']
        }))
        landmark_names.append(meta_assets[landmark['asset']]['name'])

# landmarks = config['environment']['landmarks']
# known_landmarks = []
# for landmark in landmarks:
#     if not landmark['hidden']:
#         # print(landmark)
#         known_landmarks.append(Landmark({
#             'transform': {
#                 'location': {
#                     'x': landmark['location'][0],
#                     'y': landmark['location'][1],
#                     'z': landmark['location'][2]
#                 },
#                 'rotation': {
#                     'pitch': landmark['location'][0],
#                     'yaw': landmark['location'][1],
#                     'roll': landmark['location'][2]
#                 }
#             },
#             'opendriveid': {
#                 'lane_id': landmark['lane_id'],
#                 'road_id': all_road_id[landmark['road_id']],
#                 'junction_id': all_junction_id[landmark['junction_id']]
#             },
#             'asset': landmark['asset'],
#             'name': meta_assets[landmark['asset']]['name']
#         }))
#         landmark_names.append(meta_assets[landmark['asset']]['name'])

# for landmark in known_landmarks:
#     print(landmark.to_list())

landmark_emb = landmark_encoder.forward(known_landmarks)  # [num_landmarks, lane_embedding_dim]

# TODO: Concat `landmark_emb` with `landmark_names`, dim=1
#       street_emb: [num_landmarks, id_embedding_dim]
#       landmark_names -> Text Encoder (Pre-trained): [num_landmarks, text_embedding_dim]
#       pad num_landmarks to max_num_landmarks (optional?)

print(landmark_emb.shape)
