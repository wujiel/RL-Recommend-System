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


u_m_model = UserMovieEmbedding(6041,3953, 100)
u_m_model([np.zeros((1,)),np.zeros((1,))])
u_m_model.load_weights(r'save_weights/user_movie_at_once.h5')

userlist=[2271, 349, 1626, 5614, 2424, 5675, 3240, 2731, 2587]
userlist2=[1,1,1,1,1,1,1,1,1,1,1,1]
userlist3=[1,1,1,1,1,1,1,1,1]
movielist=[1197 ,3312, 2713, 3671, 1275,   94,  655, 1573, 3638]
movielist2=[1193, 3408, 2355, 1287, 2804, 594, 919, 595, 938, 2398, 2918, 1035]
movielist3 = [661, 914, 1197, 2321, 260, 1198, 858, 1608, 1197]
lable = [1, 0, 1, 1, 1, 1, 0, 1, 0]

userndarray = np.array(userlist)
moviendarray = np.array(movielist)

userndarray2 = np.array(userlist2)
moviendarray2 = np.array(movielist2)

userndarray3 = np.array(userlist3)
moviendarray3 = np.array(movielist3)

predictions = u_m_model([userndarray,moviendarray])
print("predictions就是我：", predictions)

predictions = u_m_model([userndarray2,moviendarray2])
print("predictions就是我：", predictions)

predictions = u_m_model([userndarray3,moviendarray3])
print("predictions就是我：", predictions)