
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


users_num = 4000
items_num = 3000
embedding_dim = 100

user_id = 1
items_ids1 = [1193, 3408, 2355, 1287, 2804, 594, 919, 595, 938, 2398, 2918, 1035, 2791, 2018, 3105, 2797, 1270, 527, 48, 1097, 1721, 1545, 2294, 3186, 1566, 588, 1907, 783, 1836, 1022, 2762, 150, 1, 1961, 1962, 2692, 260, 1028, 1029, 1207, 2028, 531, 3114, 608, 1246]
items_ids2 = [661, 914, 1197, 2321, 260, 1198, 858, 1608, 1197]

srm_ave = DRRAveStateRepresentation(embedding_dim)

embedding_network = UserMovieEmbedding(users_num, items_num, embedding_dim)
user_eb = embedding_network.get_layer('user_embedding')(np.array(user_id))
items_eb1 = embedding_network.get_layer('movie_embedding')(np.array(items_ids1))
items_eb2 = embedding_network.get_layer('movie_embedding')(np.array(items_ids2))
# state = srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])


print("userembeded:",user_eb)
print("=================================================")
print("itemsembed1",items_eb1)
print(np.dot(user_eb,np.transpose(items_eb1)))
print("itemsembed2",items_eb2)
print(np.dot(user_eb,np.transpose(items_eb2)))