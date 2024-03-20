import numpy as np
import variables
import time
# import tensorflow as tf


from agent_process import AgentProcess
from multiprocessing import Process, Pipe
import threading
from collections import defaultdict, deque

class MasterProcess():
    def __init__(self, verbose=False):
        self.processes = {}
        self.data_buffer=deque(maxlen=10000)
        self.count=0
        self.Dlist=[]
        self.data=[]


    def train_agents(self):
        pipes = {}
        for i in range(4):
            parent_conn, child_conn = Pipe()
            pipes[i] = parent_conn
            print(i)
            p = AgentProcess(conn=child_conn, id=i)
            p.start()
            self.processes[i] = p


        t0 = time.time()
        def listenToAgent(id):

            while True:
                msg = pipes[id].recv()
                if msg == "saved":
                    print("Master process (0) saved his weights.")
                    for j in self.Dlist:
                        print(str(j)+" processs load")
                        self.Dlist.remove(j)
                        pipes[j].send("load")

                else:
                    print(len(msg[1]),'master')
                    id = msg[0]
                    self.Dlist.append(id)
                    print(self.Dlist)
                    self.data_buffer.extend(msg[1])
                    self.count+=1
                    print(len(self.data_buffer))
                    self.data=msg[1]
                    print("Process "+str(id)+" returns ")
                    if len(self.data_buffer)<512:
                        pipes[id].send("load")
                        pipes[0].send(["collect", msg[1]])


        threads_listen = []
        print("Threads to start")
        for id in pipes:
            t = threading.Thread(target=listenToAgent, args=(id,))
            t.start()
            threads_listen.append(t)
        print("Threads started")
        count=0

        while True:

            if len(self.data_buffer) > 512 and count!=self.count:
                count=self.count

                pipes[0].send(("train_with_batchs",self.data))



if __name__ == '__main__':
    tp = MasterProcess()#'./best_policy_torch_6_6_4.model'


    tp.train_agents()

