### pip install networkx stable-baselines3 gym torch transformers numpy
import networkx as nx
import numpy as np
import torch
from gym import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from transformers import AutoTokenizer, AutoModel, pipeline


