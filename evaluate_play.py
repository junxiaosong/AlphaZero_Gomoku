# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
from game import Board, Game
from models.mcts_alphaZero import MCTSPlayer
from models.policy_value_net_numpy import PolicyValueNetNumpy
from models.policy_value_net_pytorch import PolicyValueNet as PytorchPolicyValueNet # Pytorch
# from models.policy_value_net_tensorflow import PolicyValueNet as TensorflowPolicyValueNet# Tensorflow
from models.policy_value_net_pytorch2 import PolicyValueNet as PytorchPolicyValueNet2 # Pytorch
import pickle
import random
import os
import json
MODEL_CLASSES = {
"numpy":PolicyValueNetNumpy,
"pytorch":PytorchPolicyValueNet,
"pytorch2":PytorchPolicyValueNet2,
# "tensorflow":TensorflowPolicyValueNet,
}
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_type1", default="pytorch", type=str,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--model_type2", default="pytorch", type=str,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
parser.add_argument("--board_width", default=9,type=int, help="board_width")
parser.add_argument("--board_height",default=9,type=int,help="board_height")
parser.add_argument("--n_in_row",default=6,type=int,help="n_in_row")
parser.add_argument("--output_dir", default="./", type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_file1", default='./best_policy.model', type=str,
                    help="The model_file.")
parser.add_argument("--model_file2", default='./best_policy.model', type=str,
                    help="The model_file.")
parser.add_argument("--round_num",default=1,type=int,help="board_height")
parser.add_argument("--n_playout",default=400,type=int,help="n_playout")
parser.add_argument("--n_layer_resnet", default=-1, type=int, help="num of simulations for each move.")


args, _ = parser.parse_known_args()
print("Print the args:")
for key, value in sorted(args.__dict__.items()):
    print("{} = {}".format(key, value))

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def get_mcts_player(model_type, model_file, width, height):
    if model_type == "numpy":
        policy_param = pickle.load(open(model_file, 'rb'),
                                   encoding='bytes')  # To support python3

        best_policy = MODEL_CLASSES[model_type](args, width, height, policy_param)
    else:
        best_policy = MODEL_CLASSES[model_type](args, width, height, model_file)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                             c_puct=5,
                             n_playout=args.n_playout)  # set larger n_playout for better performance
    return mcts_player

def run():
    n = args.n_in_row
    width, height = args.board_width, args.board_height
    board = Board(width=width, height=height, n_in_row=n)
    game = Game(board)
    mcts1 = get_mcts_player(args.model_type1, args.model_file1, width, height)
    mcts2 = get_mcts_player(args.model_type2, args.model_file2, width, height)
    winner_dict = dict()
    for i in range(0, args.round_num):
        start_player =1 if random.random() > 0.5 else 0
        winner = game.start_play(mcts1, mcts2, start_player=start_player, is_shown=1)
        if winner not in winner_dict:
            winner_dict[winner] = 0
        winner_dict[winner] += 1
    print("output winner dict to {}".format(os.path.join(args.output_dir, "winner_dict.tsv")))
    with open(os.path.join(args.output_dir, "winner_dict.tsv"), 'w', encoding='utf8') as fout:
        fout.write(json.dumps(winner_dict))
    print("winner dict:")
    print(winner_dict)

if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    run()
