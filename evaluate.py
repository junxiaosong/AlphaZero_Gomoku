# -*- coding: utf-8 -*-
"""
An implementation of the evaluation pipeline of AlphaZero for Gomoku

@author: Chunlei Wang
"""

import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
#from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
#from policy_value_net_keras import PolicyValueNet # Keras
from policy_value_net_res_tensorflow import PolicyValueNetRes30 # Tensorflow
from datetime import datetime
import utils
import os

OUTPUT_DIR = "evaluation/" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
os.makedirs(OUTPUT_DIR, exist_ok=True)
EVALUATION_OUTPUT = OUTPUT_DIR + "/evaluation.txt"

class EvaluationPipeline():
    def __init__(self, current_model, baseline_model):
        # params of the board and the game
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row,
                           forbidden_hands=True)
        self.game = Game(self.board)
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5        
        
        self.baseline_policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   'l+', 
                                                   model_file=baseline_model)

        self.current_policy_value_net = PolicyValueNetRes30(self.board_width,
                                                  self.board_height,
                                                  'l+', 
                                                   model_file=current_model)

    def policy_evaluate(self, n_games=100):
        """
        Evaluate the trained policy by playing against the baseline MCTS player
        """
        current_mcts_player = MCTSPlayer(self.current_policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

        baseline_mcts_player = MCTSPlayer(self.baseline_policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          baseline_mcts_player,
                                          start_player=i % 2,
                                          is_shown=1)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        
        output = "Evaluation games: {}, num_playouts: {}, win: {}, lose: {}, tie: {}, win ratio: {}".format(
                n_games,
                self.n_playout,
                win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio)

        utils.log(output, EVALUATION_OUTPUT)
        
        return win_ratio

    def run(self):
        """run the evaluation pipeline"""
        try:
            win_ratio = self.policy_evaluate()
            return win_ratio
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    evaluation_pipeline = EvaluationPipeline(current_model='output/current_policy.model', baseline_model='output/baseline_policy.model')
    evaluation_pipeline.run()
