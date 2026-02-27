import socket
import gym
import numpy as np

class EnergyPlusEnv(gym.Env):
    def __init__(self, host="localhost", port=3000):
        super(EnergyPlusEnv, self).__init__()
        self.host = host
        self.port = port
        self.sock = None

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=np.array([18.0, 22.0]), high=np.array([24.0, 28.0]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32)

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def reset(self):
        if self.sock is None:
            self.connect()
        # Send initial action (just to trigger first step)
        init_action = "1.0 0 2 21.0 25.0\n"
        self.sock.sendall(init_action.encode())
        return self._receive_observation()

    def step(self, action):
        # Format: protocol version, error flag, number of values, then values
        msg = f"1.0 0 2 {action[0]:.2f} {action[1]:.2f}\n"
        self.sock.sendall(msg.encode())
        obs = self._receive_observation()

        # Simple reward function: keep inside temp close to 22°C
        inside_temp = obs[0]
        reward = -abs(inside_temp - 22)

        done = False  # You can add condition to end episode
        info = {}
        return obs, reward, done, info

    def _receive_observation(self):
        data = self.sock.recv(1024).decode()
        tokens = data.strip().split()
        # Example: "1.0 0 3600.0 3 22.5 10.0 1500.0"
        obs_values = list(map(float, tokens[4:]))  # Skip protocol headers
        return np.array(obs_values, dtype=np.float32)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
