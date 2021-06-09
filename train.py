# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
#from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
#from policy_value_net_keras import PolicyValueNet # Keras
from policy_value_net_res_tensorflow import PolicyValueNetRes30 # Tensorflow
from datetime import datetime
import utils
import os
import argparse

class TrainPipeline():
    def __init__(self, model_name, loss_function, forbidden_hands, init_model=None):
        # params of the board and the game
        self.board_width = 9
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row,
                           forbidden_hands=forbidden_hands)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 1000  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 3000
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.model_name = model_name
        if init_model:
            # start training from an initial policy-value net
            if self.model_name == 'baseline':
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                    self.board_height,
                                                    loss_function,
                                                    model_file=init_model)
            else:
                self.policy_value_net = PolicyValueNetRes30(self.board_width,
                                                            self.board_height,
                                                            loss_function,
                                                            model_file=init_model)
        else:
            # start training from a new policy-value net
            if self.model_name == 'baseline':
                self.policy_value_net = PolicyValueNet(self.board_width,
                                                    self.board_height,
                                                    loss_function)
            else:
                self.policy_value_net = PolicyValueNetRes30(self.board_width,
                                                    self.board_height,
                                                    loss_function)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                        self.model_name,
                                                        temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self, batch_num, episode_len):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        utils.log(("batch:{},"
                "episode_len:{},"
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
                ).format(batch_num,
                        episode_len,
                        kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new), INTERMEDIATE_RESULT)
        return loss, entropy

    def policy_evaluate(self, current_batch, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        
        output = "current self play batch: {}, num_playouts: {}, win: {}, lose: {}, tie: {}, win ratio: {}".format(
                current_batch,
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1], win_ratio)

        utils.log(output, SCORE_OUTPUT)
        
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update(i+1, self.episode_len)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    utils.log("current self-play batch: {}".format(i+1), CONSOLE_OUTPUT)
                    win_ratio = self.policy_evaluate(current_batch=i+1)
                    self.policy_value_net.save_model(OUTPUT_DIR+'/current_policy.model')
                    if win_ratio >= self.best_win_ratio:
                        utils.log("New best policy!!!!!!!!", CONSOLE_OUTPUT)
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(OUTPUT_DIR+'/best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--ModelName', '-m', dest='ModelName', required=True, choices=['baseline', 'res30'])
    parser.add_argument('--LossFunction', '-l', dest='LossFunction', required=True, choices=['lv', 'lp', 'l+', 'lx'])
    parser.add_argument('--EnableForbiddenHands', '-fh', dest='EnableForbiddenHands', action='store_false', help=r'Enable forbidden hands')

    args = parser.parse_args()
    model_name = args.ModelName
    loss_function = args.LossFunction
    forbidden_hands = args.EnableForbiddenHands
    
    OUTPUT_DIR = "output/" + "baseline" if model_name == "baseline" else "res30"
    OUTPUT_DIR += "_forbiddenhands/" if forbidden_hands else "/"
    init_model = OUTPUT_DIR + "current_policy.model"
    if not os.path.exists(init_model):
        init_model = None
    OUTPUT_DIR += datetime.utcnow().strftime("%Y%m%d%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    INTERMEDIATE_RESULT = OUTPUT_DIR + "/intermediate_result.txt"
    SCORE_OUTPUT = OUTPUT_DIR + "/scores.txt"
    CONSOLE_OUTPUT = OUTPUT_DIR + "/console.txt"

    print("**************************************************************")
    print("Start new training process...")
    print(f"ModelName: {model_name}, LossFunction: {loss_function}, EnableForbiddenHands: {forbidden_hands}")
    print(f"init model : {init_model}")
    print(f"intermediate result : {INTERMEDIATE_RESULT}")
    print(f"score output : {SCORE_OUTPUT}")
    print(f"console output : {CONSOLE_OUTPUT}")
    print("**************************************************************")

    training_pipeline = TrainPipeline(model_name, loss_function, forbidden_hands, init_model)
    training_pipeline.run()
