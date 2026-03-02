import os
import zmq
import json
import numpy as np

class DataPublisher(object):
    def __init__(self, port: int = None) -> None:
        context = zmq.Context()
        self.serverSocket = context.socket(zmq.PUB)
        env_port = os.environ.get("ZMQ_PUB_PORT")
        port = int(env_port) if env_port else port
        if port is None:
            port = 9872
        try:
            self.serverSocket.bind("tcp://*:" + str(port))
            self.port = port
        except zmq.ZMQError:
            self.port = self.serverSocket.bind_to_random_port("tcp://*")
        self.data = {"A_timestamp": 0, }

    def step_publisher(self, t):
        self.data["A_timestamp"] = t
        self.serverSocket.send_string(json.dumps(self.data))

    def pub_vector(self, name, vec: np.ndarray):
        for k in range(len(vec)):
            self.data[name + str(k)] = vec[k]