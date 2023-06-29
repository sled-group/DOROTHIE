import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from glob import glob
import json
import shutil

from tqdm import tqdm

from data.dataset import Game

def main():
    parser = ArgumentParser(description='TOTO')

    parser.add_argument('--dataset_path', type=str, default='SDN',
                    help='Path to dataset')

    parser.add_argument('--map_path', type=str, default='data/towns',
                    help='Path to map data')
    parser.add_argument('--buffer_size', type=int, default=100,
                    help='length of image buffer')
    parser.add_argument('--buffer_step', type=int, default=4,
                    help='step of image buffer')
    parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use')
    parser.add_argument('--ablation', type=int, default=0,
                    help='Which ablation to run sequentially [0..7 or None]')


    args = parser.parse_args()

    game_paths = [x.split('/')[-1] for x in sorted(glob(f'{args.dataset_path}/*')) if os.path.isdir(x)]
    # exit(0)
    with open("final.json") as f:
        game_paths=json.load(f)
    print(game_paths)

    for i   in tqdm(range(0,len(game_paths))):
        game_path= game_paths[i]
        # print(i,game_path)
        # annotate_path= annotate_paths[i]
        # os.mkdir(annotate_path)
        # shutil.copy(os.path.join(game_path,"config.json"),os.path.join(annotate_path,"config.json"))
        # shutil.copy(os.path.join(game_path,"plan.json"),os.path.join(annotate_path,"plan.json"))
        # shutil.copy(os.path.join(game_path,"annotated_log.json"),os.path.join(annotate_path,"annotated_log.json"))
        # shutil.copy(os.path.join(game_path,"trajectory.csv"),os.path.join(annotate_path,"trajectory.csv"))
        # shutil.copytree(os.path.join(game_path,"utterance"),os.path.join(annotate_path,"utterance"))
        game = Game(game_path,args,use_cache=False)
        # for event in game.output_events:
        #     if event[]


if __name__ == "__main__":
    main()