
# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent

import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 10

# Loading datasets
ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR ,'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR ,'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR ,'movies.dat') ,encoding='latin-1').readlines()]
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)



# 영화 id를 영화 제목으로j
movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}



len(movies_list)



ratings_df.head(5)



# # 사용자가 선택한 영화 분포 확인
# plt.figure(figsize=(20,10))
# plt.hist(ratings_df["MovieID"], bins=3883)
# plt.show()



# 결측치 확인
ratings_df.isnull().sum()



# 최대값 확인
print(len(set(ratings_df["UserID"])) == max([int(i) for i in set(ratings_df["UserID"])]))
print(max([int(i) for i in set(ratings_df["UserID"])]))



ratings_df = ratings_df.applymap(int)



# 유저별로 본 영화들 순서대로 정리
users_dict = {user : [] for user in set(ratings_df["UserID"])}
users_dict[1]



# 시간 순으로 정렬하기
ratings_df = ratings_df.sort_values(by='Timestamp', ascending=True)
ratings_df.head(5)



# 유저 딕셔너리에 (영화, 평점)쌍 넣기
# 각 유저별 영화 히스토리 길이를 평점 4이상인 영화만 카운트
ratings_df_gen = ratings_df.iterrows()
users_dict_for_history_len = {user : [] for user in set(ratings_df["UserID"])}
for data in ratings_df_gen:
    users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
    if data[1]['Rating'] >= 4:
        users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))



# 각 유저별 영화 히스토리 길이
users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_df["UserID"])]



len(users_history_lens)



users_dict[1]



np.save("./data/user_dict.npy", users_dict)
np.save("./data/users_histroy_len.npy", users_history_lens)



users_num = max(ratings_df["UserID"] ) +1
items_num = max(ratings_df["MovieID"] ) +1



print(users_num, items_num)



### Training setting



train_users_num = int(users_num * 0.8)
train_items_num = items_num
print(train_users_num, train_items_num)



train_users_dict = {k :users_dict[k] for k in range(1, train_users_num +1)}
train_users_history_lens = users_history_lens[:train_users_num]
print(len(train_users_dict) ,len(train_users_history_lens))



### Evaluating setting



eval_users_num = int(users_num * 0.2)
eval_items_num = items_num
print(eval_users_num, eval_items_num)



eval_users_dict = {k :users_dict[k] for k in range(users_num -eval_users_num, users_num)}
eval_users_history_lens = users_history_lens[-eval_users_num:]
print(len(eval_users_dict) ,len(eval_users_history_lens))



### 준비된것
# users_dict, users_history_len, movies_id_to_movies, sers_num, items_num



### Evalutation



def evaluate(recommender, env, check_movies = False, top_k=False):

    # episodic reward 리셋
    episode_reward = 0
    steps = 0
    mean_precision = 0
    mean_ndcg = 0
    # Environment 리셋
    user_id, items_ids, done = env.reset()
    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')
        print('items : \n', np.array(env.get_items_names(items_ids)))

    while not done:

        # Observe current state & Find action
        ## Embedding 해주기
        user_eb = recommender.embedding_network.get_layer('user_embedding')(np.array(user_id))
        items_eb = recommender.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
        ## SRM으로 state 출력
        state = recommender.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])
        ## Action(ranking score) 출력
        action = recommender.actor.network(state)
        ## Item 추천
        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)
        if check_movies:
            print(f'recommended items ids : {recommended_item}')
            print(f'recommened items : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')
        # Calculate reward & observe new state (in env)
        ## Step
        next_items_ids, reward, done, = env.step(recommended_item, top_k=top_k)
        if top_k:
            correct_list = [1 if r > 0 else 0 for r in reward]
            # ndcg
            dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])
            mean_ndcg += dcg /idcg

            # precision
            correct_num = top_k -correct_list.count(0)
            mean_precision += correct_num /top_k

        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1

        if check_movies:
            print \
                (f'precision : {correct_num /top_k}, dcg : {dcg:0.3f}, idcg : {idcg:0.3f}, ndcg : {dcg /idcg:0.3f}, reward : {reward}')
            print()
        break

    if check_movies:
        print(f'precision : {mean_precision/steps}, ngcg : {mean_ndcg /steps}, episode_reward : {episode_reward}')
        print()

    return mean_precision /steps, mean_ndcg /steps

def calculate_ndcg(rel, irel):
    dcg = 0
    idcg = 0
    rel = [1 if r> 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r) / np.log2(i + 2)
        idcg += (ir) / np.log2(i + 2)
    return dcg, idcg


tf.keras.backend.set_floatx('float64')

sum_precision = 0
sum_ndcg = 0
TOP_K = 10

for user_id in eval_users_dict.keys():
    env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, STATE_SIZE, fix_user_id=user_id)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.load_model(
        r'save_weights\actor_8000_fixed.h5',
        r'save_weights\critic_8000_fixed.h5')
    precision, ndcg = evaluate(recommender, env, top_k=TOP_K)
    sum_precision += precision
    sum_ndcg += ndcg

print(f'precision@{TOP_K} : {sum_precision / len(eval_users_dict)}, ndcg@{TOP_K} : {sum_ndcg / len(eval_users_dict)}')














