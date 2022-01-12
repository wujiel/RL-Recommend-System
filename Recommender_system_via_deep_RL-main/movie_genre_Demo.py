from replay_buffer import PriorityExperienceReplay
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp

from actor import Actor
from critic import Critic
from replay_memory import ReplayMemory
from embedding import MovieGenreEmbedding, UserMovieEmbedding
from state_representation import DRRAveStateRepresentation

import matplotlib.pyplot as plt
import wandb


m_g_model = MovieGenreEmbedding(3953, 19, 100)
m_g_model([np.zeros((1,)),np.zeros((1,))])
m_g_model.load_weights(r'save_weights/m_g_model_weights.h5')
items_ebs =m_g_model.get_layer('movie_embedding')(466)

movielist=[466, 2564, 2767, 2772, 3810, 1856, 3531]
genrelist=[0, 2,  6, 11, 15,  7, 11]
lable = [1, 0, 0, 0, 1, 0, 0]
moviendarray = np.array(movielist)
genrendarray = np.array(genrelist)

predictions = m_g_model([moviendarray,genrendarray])
print("predictions就是我：", predictions)
