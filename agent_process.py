from multiprocessing import Process, Queue

import variables
import threading
import time

import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer

import time
import os
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow



class AgentProcess(Process):
    def __init__(self, conn,id):
        super(AgentProcess,self).__init__()
        self.conn = conn
        self.id = id
        self.msg_queue = []
        np.random.seed(self.id*100)

        self.temp = 1.0
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.epochs = 50
        self.batch_size = 512
        self.kl_targ = 0.02
        self.n_in_row = 5
        self.n_playout = 600  # num of simulations for each move 深度mcst模拟次数
        self.c_puct = 5
        self.best_win_ratio=0.0


        self.board = Board(width=variables.board_width,
                           height=variables.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.pure_mcts_playout_num=1000

        print(self.game)



    def run(self):

        buffer_size = 10000
        data_buffer = deque(maxlen=buffer_size)

        self.agent = PolicyValueNet(board_width=variables.board_width,board_height=variables.board_height,model_file=variables.init_model)

        self.count=0



        mcts_player = MCTSPlayer(self.agent.policy_value_fn,
                                     c_puct=self.c_puct,
                                     n_playout=self.n_playout,
                                     is_selfplay=1)

        def collect_selfplay_data():
            """collect self-play data for training"""

            a = time.time()
            winner, play_data = self.game.start_self_play(mcts_player,
                                                          temp=self.temp, is_shown=0)
            print(time.time() - a, 'game play')
            play_data = list(play_data)[:]
            episode_len = len(play_data)
            # augment the data
            play_data = get_equi_data(play_data)

            return play_data

        def get_equi_data(play_data):
            """augment the data set by rotation and flipping
            play_data: [(state, mcts_prob, winner_z), ..., ...]
            """
            extend_data = []
            for state, mcts_porb, winner in play_data:
                for i in [1, 2, 3, 4]:
                    # rotate counterclockwise
                    equi_state = np.array([np.rot90(s, i) for s in state])
                    equi_mcts_prob = np.rot90(np.flipud(
                        mcts_porb.reshape(variables.board_height, variables.board_width)), i)
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

        def policy_update(data_buffer):
            """update the policy-value net"""

            mini_batch = random.sample(data_buffer, self.batch_size)
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]
            old_probs, old_v = self.agent.policy_value(state_batch)
            for i in range(self.epochs):
                loss, entropy = self.agent.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate * self.lr_multiplier)
                new_probs, new_v = self.agent.policy_value(state_batch)
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
            print(("kl:{:.5f},"
                   "lr_multiplier:{:.3f},"
                   "loss:{:.3f},"
                   "entropy:{:.3f},"
                   "explained_var_old:{:.3f},"
                   "explained_var_new:{:.3f}"
                   ).format(kl,
                            self.lr_multiplier,
                            loss,
                            entropy,
                            explained_var_old,
                            explained_var_new))
            self.agent.save_model(variables.init_model)
            # modelfile = './current_policy.model'
            # return modelfile

        def policy_evaluate(n_games=10):
            """
            Evaluate the trained policy by playing against the pure MCTS player
            Note: this is only for monitoring the progress of training
            """
            print('eval')
            current_mcts_player = MCTSPlayer(self.agent.policy_value_fn,c_puct=self.c_puct,n_playout=self.n_playout)
            pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
            win_cnt = defaultdict(int)
            print('eval run')
            for i in range(n_games):
                winner = self.game.start_play(current_mcts_player,
                                              pure_mcts_player,
                                              start_player=i % 2,
                                              is_shown=0)
                win_cnt[winner] += 1
            win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
            print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
            return win_ratio


        def treatQueue():
            print('treatQueue In ' + str(os.getpid()))
            t0 = time.time()
            try:
                msg = self.conn.recv()
            except Exception as e:
                msg = self.conn.recv()

                print(str(e)+" "+str(self.id)+" "+str(os.getpid()))
            if msg == "load":
                print(str(os.getpid())+' start load')
                self.agent.restore_model(variables.init_model)
                print("Process "+str(os.getpid())+" loaded the master (0) model.")

            elif msg[0] == "collect":
                data_buffer.extend(msg[1])
                print(len(msg[1]), len(data_buffer))

            elif msg[0] == "train_with_batchs":
                self.count += 1
                print("Master process is training ... "+str(self.count))

                data_buffer.extend(msg[1])
                print(len(msg[1]),len(data_buffer))
                policy_update(data_buffer)
                self.agent.save_model(variables.init_model)
                print("Master process finished training. Time : "+str(time.time()-t0)+" \n")

                if self.count% variables.check_freq == 0:
                    print("current self-play batch: {}".format(self.count))
                    win_ratio = policy_evaluate()
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.agent.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                self.conn.send("saved")

            print('treatQueue Out '+ str(os.getpid()))

        while True:
            if self.id!= 0:
                playdata=collect_selfplay_data()
                print("Process "+str(self.id)+" finished playing."+str(len(playdata)))
                self.conn.send([self.id,playdata])
            treatQueue()
